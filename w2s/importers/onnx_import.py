"""
onnx_import.py — Load ONNX models into the w2s ComputeGraph IR.

Usage:
    from w2s.importers.onnx_import import load_onnx
    graph = load_onnx("model.onnx")
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from w2s.core import ComputeGraph, OpType, Operation

# ---------------------------------------------------------------------------
#  Lazy onnx import — raise a clear error if the package isn't installed.
# ---------------------------------------------------------------------------

try:
    import onnx
    from onnx import numpy_helper, shape_inference, TensorProto
except ImportError:
    raise ImportError(
        "The 'onnx' package is required by w2s.importers.onnx_import.\n"
        "Install it with:  pip install onnx"
    )


# ---------------------------------------------------------------------------
#  ONNX op-type -> w2s OpType mapping
# ---------------------------------------------------------------------------

_OPTYPE_MAP: Dict[str, OpType] = {
    "Gemm": OpType.DENSE,
    "MatMul": OpType.DENSE,
    "Conv": OpType.CONV2D,            # refined to CONV1D based on weight dims
    "Relu": OpType.RELU,
    "Sigmoid": OpType.SIGMOID,
    "Tanh": OpType.TANH,
    "Gelu": OpType.GELU,
    "Add": OpType.ADD,
    "Mul": OpType.MULTIPLY,
    "LayerNormalization": OpType.LAYERNORM,
    "BatchNormalization": OpType.BATCHNORM,
    "Softmax": OpType.SOFTMAX,
    "MaxPool": OpType.MAXPOOL2D,
    "AveragePool": OpType.AVGPOOL2D,
    "GlobalAveragePool": OpType.GLOBAL_AVGPOOL,
    "Reshape": OpType.RESHAPE,
    "Flatten": OpType.FLATTEN,
    "Concat": OpType.CONCAT,
    "Gather": OpType.EMBEDDING,
}


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _onnx_attr(node, name: str, default=None):
    """Extract a named attribute from an ONNX node."""
    for attr in node.attribute:
        if attr.name == name:
            if attr.type == onnx.AttributeProto.INT:
                return attr.i
            elif attr.type == onnx.AttributeProto.INTS:
                return tuple(attr.ints)
            elif attr.type == onnx.AttributeProto.FLOAT:
                return attr.f
            elif attr.type == onnx.AttributeProto.FLOATS:
                return tuple(attr.floats)
            elif attr.type == onnx.AttributeProto.STRING:
                return attr.s.decode("utf-8")
            elif attr.type == onnx.AttributeProto.TENSOR:
                return numpy_helper.to_array(attr.t)
            else:
                return default
    return default


def _collect_initializers(graph) -> Dict[str, np.ndarray]:
    """Return a dict mapping initializer name -> numpy array."""
    inits: Dict[str, np.ndarray] = {}
    for tensor in graph.initializer:
        inits[tensor.name] = numpy_helper.to_array(tensor)
    return inits


def _extract_shape(type_proto) -> Optional[Tuple[int, ...]]:
    """Try to extract a static shape from an ONNX TypeProto."""
    if not type_proto.HasField("tensor_type"):
        return None
    shape_proto = type_proto.tensor_type.shape
    if shape_proto is None:
        return None
    dims: List[int] = []
    for d in shape_proto.dim:
        if d.dim_param:
            dims.append(-1)              # dynamic / symbolic
        else:
            dims.append(d.dim_value)
    return tuple(dims)


def _unique_name(base: str, used: Set[str]) -> str:
    """Generate a unique name, appending _1, _2, ... if needed."""
    if base not in used:
        used.add(base)
        return base
    i = 1
    while f"{base}_{i}" in used:
        i += 1
    name = f"{base}_{i}"
    used.add(name)
    return name


def _sanitize(name: str) -> str:
    """Turn an ONNX name into a valid, short identifier."""
    name = name.replace("/", "_").replace(":", "_").replace(".", "_")
    if name.startswith("_"):
        name = "t" + name
    return name


# ---------------------------------------------------------------------------
#  Attribute builders per op type
# ---------------------------------------------------------------------------

def _build_attrs_conv(node, inits: Dict[str, np.ndarray]) -> Dict[str, Any]:
    kernel_shape = _onnx_attr(node, "kernel_shape", ())
    strides = _onnx_attr(node, "strides", (1,) * len(kernel_shape))
    pads = _onnx_attr(node, "pads", (0,) * (2 * len(kernel_shape)))
    group = _onnx_attr(node, "group", 1)
    # pads is [top, left, bottom, right] or [before, after] — collapse to
    # (pad_h, pad_w) by taking the first half.
    half = len(pads) // 2
    padding = tuple(pads[:half])
    return {
        "kernel_size": tuple(kernel_shape),
        "stride": tuple(strides),
        "padding": padding,
        "groups": group,
    }


def _build_attrs_pool(node) -> Dict[str, Any]:
    kernel_shape = _onnx_attr(node, "kernel_shape", (2, 2))
    strides = _onnx_attr(node, "strides", kernel_shape)
    return {
        "kernel_size": tuple(kernel_shape),
        "stride": tuple(strides),
    }


def _build_attrs_layernorm(node) -> Dict[str, Any]:
    eps = _onnx_attr(node, "epsilon", 1e-5)
    axis = _onnx_attr(node, "axis", -1)
    return {"eps": eps, "axis": axis}


def _build_attrs_batchnorm(node) -> Dict[str, Any]:
    eps = _onnx_attr(node, "epsilon", 1e-5)
    momentum = _onnx_attr(node, "momentum", 0.9)
    return {"eps": eps, "momentum": momentum}


def _build_attrs_concat(node) -> Dict[str, Any]:
    axis = _onnx_attr(node, "axis", 0)
    return {"axis": axis}


def _build_attrs_softmax(node) -> Dict[str, Any]:
    axis = _onnx_attr(node, "axis", -1)
    return {"axis": axis}


def _build_attrs_reshape(node, inits: Dict[str, np.ndarray]) -> Dict[str, Any]:
    # The target shape may come from a constant initializer input.
    if len(node.input) >= 2 and node.input[1] in inits:
        target_shape = tuple(int(x) for x in inits[node.input[1]])
        return {"target_shape": target_shape}
    return {}


def _build_attrs_flatten(node) -> Dict[str, Any]:
    axis = _onnx_attr(node, "axis", 1)
    return {"axis": axis}


def _build_attrs_embedding(node, inits: Dict[str, np.ndarray]) -> Dict[str, Any]:
    # Gather used as embedding lookup: input 0 is the weight table.
    weight_name = node.input[0] if len(node.input) > 0 else None
    if weight_name and weight_name in inits:
        w = inits[weight_name]
        return {"num_embeddings": w.shape[0], "embedding_dim": w.shape[1]}
    return {}


def _build_attrs_gemm(node) -> Dict[str, Any]:
    alpha = _onnx_attr(node, "alpha", 1.0)
    beta = _onnx_attr(node, "beta", 1.0)
    transA = _onnx_attr(node, "transA", 0)
    transB = _onnx_attr(node, "transB", 0)
    return {"alpha": alpha, "beta": beta, "transA": transA, "transB": transB}


# ---------------------------------------------------------------------------
#  Fusion pass: detect simple op pairs (e.g., Conv+Relu, Gemm+Relu)
# ---------------------------------------------------------------------------

def _fuse_activations(nodes: list, inits: Dict[str, np.ndarray]):
    """
    One-pass fusion: when a producer (Conv/Gemm/MatMul) has exactly one
    consumer and that consumer is Relu, fold the activation into the
    producer's attrs and delete the consumer.

    Returns (new_node_list, output_renames) where output_renames maps the
    old consumer output to the producer's output (since the consumer is gone).
    """
    # Map: tensor_name -> list of consumer nodes
    consumers: Dict[str, List] = {}
    for n in nodes:
        for inp in n.input:
            consumers.setdefault(inp, []).append(n)

    fusable_producers = {"Conv", "Gemm", "MatMul"}
    fusable_acts = {"Relu": "relu", "Sigmoid": "sigmoid", "Tanh": "tanh"}

    fused_set: Set[str] = set()          # names of deleted activation nodes
    output_renames: Dict[str, str] = {}   # old act output -> producer output

    for n in nodes:
        if n.op_type not in fusable_producers:
            continue
        if len(n.output) != 1:
            continue
        out_name = n.output[0]
        users = consumers.get(out_name, [])
        if len(users) != 1:
            continue
        act_node = users[0]
        if act_node.op_type not in fusable_acts:
            continue
        # Fuse!
        act_str = fusable_acts[act_node.op_type]
        # We'll record this; the attribute is attached later when building
        # the Operation.
        n.doc_string = f"__fused_activation__={act_str}"
        # Map the activation's output to the producer's output.
        if len(act_node.output) == 1:
            output_renames[act_node.output[0]] = out_name
        fused_set.add(act_node.name or id(act_node))

    # Build a new node list without the fused activation nodes.
    new_nodes = []
    for n in nodes:
        key = n.name or id(n)
        if key not in fused_set:
            new_nodes.append(n)

    return new_nodes, output_renames


# ---------------------------------------------------------------------------
#  Weight extraction helpers
# ---------------------------------------------------------------------------

def _extract_weights_conv(node, inits: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    weights: Dict[str, np.ndarray] = {}
    if len(node.input) >= 2 and node.input[1] in inits:
        weights["weight"] = inits[node.input[1]]
    if len(node.input) >= 3 and node.input[2] in inits:
        weights["bias"] = inits[node.input[2]]
    return weights


def _extract_weights_gemm(node, inits: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    weights: Dict[str, np.ndarray] = {}
    # Gemm: Y = alpha * A * B + beta * C
    # Typically input 0 is activation, input 1 is weight, input 2 is bias.
    if len(node.input) >= 2 and node.input[1] in inits:
        w = inits[node.input[1]]
        transB = _onnx_attr(node, "transB", 0)
        if transB:
            w = w.T
        weights["weight"] = w
    if len(node.input) >= 3 and node.input[2] in inits:
        weights["bias"] = inits[node.input[2]]
    return weights


def _extract_weights_matmul(node, inits: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    weights: Dict[str, np.ndarray] = {}
    # Either input could be the weight matrix.
    if node.input[1] in inits:
        weights["weight"] = inits[node.input[1]]
    elif node.input[0] in inits:
        weights["weight"] = inits[node.input[0]]
    return weights


def _extract_weights_layernorm(node, inits: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    weights: Dict[str, np.ndarray] = {}
    if len(node.input) >= 2 and node.input[1] in inits:
        weights["scale"] = inits[node.input[1]]
    if len(node.input) >= 3 and node.input[2] in inits:
        weights["bias"] = inits[node.input[2]]
    return weights


def _extract_weights_batchnorm(node, inits: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    weights: Dict[str, np.ndarray] = {}
    # BatchNorm: X, scale, B, input_mean, input_var
    names = ["scale", "bias", "running_mean", "running_var"]
    for i, key in enumerate(names, start=1):
        if len(node.input) > i and node.input[i] in inits:
            weights[key] = inits[node.input[i]]
    return weights


def _extract_weights_embedding(node, inits: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    weights: Dict[str, np.ndarray] = {}
    # Gather: input 0 is the embedding table.
    if len(node.input) >= 1 and node.input[0] in inits:
        weights["weight"] = inits[node.input[0]]
    return weights


# ---------------------------------------------------------------------------
#  Public entry point
# ---------------------------------------------------------------------------

def load_onnx(path: str, name: str = None) -> ComputeGraph:
    """Load an ONNX model and convert to a ComputeGraph.

    Parameters
    ----------
    path : str
        Filesystem path to an ``.onnx`` file.
    name : str, optional
        Name for the resulting graph.  Defaults to the filename stem.

    Returns
    -------
    ComputeGraph
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"ONNX model not found: {path}")

    model = onnx.load(path)

    # Run shape inference so value_info is populated.
    try:
        model = shape_inference.infer_shapes(model)
    except Exception:
        pass  # best-effort; some custom ops may block this

    g = model.graph
    if name is None:
        name = os.path.splitext(os.path.basename(path))[0]

    inits = _collect_initializers(g)
    init_names: Set[str] = set(inits.keys())

    # Collect value_info shapes (inputs + intermediates + outputs).
    shape_map: Dict[str, Tuple[int, ...]] = {}
    for vi in list(g.input) + list(g.value_info) + list(g.output):
        s = _extract_shape(vi.type)
        if s is not None:
            shape_map[vi.name] = s

    # --- Fusion pass ---
    nodes = list(g.node)
    nodes, output_renames = _fuse_activations(nodes, inits)

    # Build a helper to apply renames transitively.
    def _resolve(tensor: str) -> str:
        seen: Set[str] = set()
        while tensor in output_renames:
            if tensor in seen:
                break
            seen.add(tensor)
            tensor = output_renames[tensor]
        return tensor

    # --- Identify true graph inputs (not initializers) ---
    graph_input_names: List[str] = []
    graph_input_shapes: Dict[str, Tuple[int, ...]] = {}
    for inp in g.input:
        if inp.name not in init_names:
            graph_input_names.append(inp.name)
            s = _extract_shape(inp.type)
            if s is not None:
                graph_input_shapes[inp.name] = s

    graph_output_names: List[str] = [_resolve(o.name) for o in g.output]

    # --- Build operations ---
    used_names: Set[str] = set()
    operations: List[Operation] = []

    for node in nodes:
        onnx_op = node.op_type
        if onnx_op not in _OPTYPE_MAP:
            # Skip ops we don't handle (Constant, Identity, etc.)
            continue

        op_type = _OPTYPE_MAP[onnx_op]

        # Refine Conv1D vs Conv2D based on weight rank.
        if onnx_op == "Conv":
            w_name = node.input[1] if len(node.input) >= 2 else None
            if w_name and w_name in inits and inits[w_name].ndim == 3:
                op_type = OpType.CONV1D

        # Name
        raw_name = node.name or f"{op_type.value}_{len(operations)}"
        op_name = _unique_name(_sanitize(raw_name), used_names)

        # Inputs — filter out initializer-only inputs (already in weights)
        op_inputs: List[str] = []
        for inp in node.input:
            if inp == "":
                continue
            resolved = _resolve(inp)
            if resolved not in init_names:
                op_inputs.append(resolved)

        # Outputs
        op_outputs: List[str] = [_resolve(o) for o in node.output if o != ""]

        # Attributes
        attrs: Dict[str, Any] = {}
        if onnx_op == "Conv":
            attrs = _build_attrs_conv(node, inits)
        elif onnx_op in ("MaxPool", "AveragePool"):
            attrs = _build_attrs_pool(node)
        elif onnx_op == "LayerNormalization":
            attrs = _build_attrs_layernorm(node)
        elif onnx_op == "BatchNormalization":
            attrs = _build_attrs_batchnorm(node)
        elif onnx_op == "Concat":
            attrs = _build_attrs_concat(node)
        elif onnx_op == "Softmax":
            attrs = _build_attrs_softmax(node)
        elif onnx_op == "Reshape":
            attrs = _build_attrs_reshape(node, inits)
        elif onnx_op == "Flatten":
            attrs = _build_attrs_flatten(node)
        elif onnx_op == "Gather":
            attrs = _build_attrs_embedding(node, inits)
        elif onnx_op == "Gemm":
            attrs = _build_attrs_gemm(node)

        # Check for fused activation (set during fusion pass).
        fused_act = None
        if node.doc_string and node.doc_string.startswith("__fused_activation__="):
            fused_act = node.doc_string.split("=", 1)[1]
            attrs["activation"] = fused_act

        # Weights
        weights: Dict[str, np.ndarray] = {}
        if onnx_op == "Conv":
            weights = _extract_weights_conv(node, inits)
        elif onnx_op == "Gemm":
            weights = _extract_weights_gemm(node, inits)
        elif onnx_op == "MatMul":
            weights = _extract_weights_matmul(node, inits)
        elif onnx_op == "LayerNormalization":
            weights = _extract_weights_layernorm(node, inits)
        elif onnx_op == "BatchNormalization":
            weights = _extract_weights_batchnorm(node, inits)
        elif onnx_op == "Gather":
            weights = _extract_weights_embedding(node, inits)

        operations.append(Operation(
            op_type=op_type,
            name=op_name,
            inputs=op_inputs,
            outputs=op_outputs,
            attrs=attrs,
            weights=weights,
        ))

    return ComputeGraph(
        name=name,
        operations=operations,
        input_names=graph_input_names,
        input_shapes=graph_input_shapes,
        output_names=graph_output_names,
    )
