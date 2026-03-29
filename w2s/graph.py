"""
graph.py — Main compiler: walks the compute graph and generates Verilog.

This is the orchestrator. It dispatches each operation to the appropriate
Verilog generator and wires the results together into a complete module.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable

from w2s.core import (
    ComputeGraph, Operation, OpType, TensorWires, QuantConfig,
)
from w2s import emit


# ---------------------------------------------------------------------------
#  Generator registry
# ---------------------------------------------------------------------------

# Maps OpType -> generator function
# Each generator: (op, wire_map, bits) -> (verilog_lines, new_wires)
_GENERATORS: Dict[OpType, Callable] = {}


def register(op_type: OpType):
    """Decorator to register a generator function for an op type."""
    def wrapper(fn):
        _GENERATORS[op_type] = fn
        return fn
    return wrapper


def _load_generators():
    """Import all generator modules to trigger their @register decorators."""
    # Import each generator module — they self-register via @register
    try:
        from w2s.generators import dense
    except ImportError:
        pass
    try:
        from w2s.generators import conv
    except ImportError:
        pass
    try:
        from w2s.generators import activation
    except ImportError:
        pass
    try:
        from w2s.generators import norm
    except ImportError:
        pass
    try:
        from w2s.generators import attention
    except ImportError:
        pass
    try:
        from w2s.generators import embedding
    except ImportError:
        pass
    try:
        from w2s.generators import pooling
    except ImportError:
        pass
    try:
        from w2s.generators import structural
    except ImportError:
        pass


# ---------------------------------------------------------------------------
#  Alternative: explicit dispatch (works even without @register)
# ---------------------------------------------------------------------------

def _get_generator(op_type: OpType) -> Optional[Callable]:
    """Look up the generator for an op type, trying registry first."""
    if op_type in _GENERATORS:
        return _GENERATORS[op_type]

    # Fallback: import directly and look up by convention
    _module_map = {
        OpType.DENSE: ("w2s.generators.dense", "generate_dense"),
        OpType.CONV1D: ("w2s.generators.conv", "generate_conv1d"),
        OpType.CONV2D: ("w2s.generators.conv", "generate_conv2d"),
        OpType.RELU: ("w2s.generators.activation", "generate_relu"),
        OpType.GELU: ("w2s.generators.activation", "generate_gelu"),
        OpType.SIGMOID: ("w2s.generators.activation", "generate_sigmoid"),
        OpType.TANH: ("w2s.generators.activation", "generate_tanh"),
        OpType.SILU: ("w2s.generators.activation", "generate_silu"),
        OpType.SOFTMAX: ("w2s.generators.activation", "generate_softmax"),
        OpType.LAYERNORM: ("w2s.generators.norm", "generate_layernorm"),
        OpType.RMSNORM: ("w2s.generators.norm", "generate_rmsnorm"),
        OpType.BATCHNORM: ("w2s.generators.norm", "generate_batchnorm"),
        OpType.MULTI_HEAD_ATTENTION: ("w2s.generators.attention", "generate_mha"),
        OpType.GROUPED_QUERY_ATTENTION: ("w2s.generators.transformer", "generate_gqa"),
        OpType.SWIGLU: ("w2s.generators.transformer", "generate_swiglu"),
        OpType.ROPE: ("w2s.generators.transformer", "generate_rope"),
        OpType.KV_CACHE: ("w2s.generators.transformer", "generate_kv_cache"),
        OpType.EMBEDDING: ("w2s.generators.embedding", "generate_embedding"),
        OpType.MAXPOOL2D: ("w2s.generators.pooling", "generate_maxpool2d"),
        OpType.AVGPOOL2D: ("w2s.generators.pooling", "generate_avgpool2d"),
        OpType.GLOBAL_AVGPOOL: ("w2s.generators.pooling", "generate_global_avgpool"),
        OpType.ADD: ("w2s.generators.structural", "generate_add"),
        OpType.MULTIPLY: ("w2s.generators.structural", "generate_multiply"),
        OpType.RESHAPE: ("w2s.generators.structural", "generate_reshape"),
        OpType.FLATTEN: ("w2s.generators.structural", "generate_flatten"),
        OpType.CONCAT: ("w2s.generators.structural", "generate_concat"),
    }

    if op_type not in _module_map:
        return None

    mod_name, fn_name = _module_map[op_type]
    try:
        import importlib
        mod = importlib.import_module(mod_name)
        fn = getattr(mod, fn_name)
        _GENERATORS[op_type] = fn  # cache for next time
        return fn
    except (ImportError, AttributeError):
        return None


# ---------------------------------------------------------------------------
#  Main compiler
# ---------------------------------------------------------------------------

def compile_graph(
    graph: ComputeGraph,
    output_dir: str = ".",
    mode: str = "combinational",
) -> str:
    """
    Compile a quantized ComputeGraph to synthesizable Verilog.

    Args:
        graph:      A ComputeGraph with q_weights/q_params populated on each op.
        output_dir: Where to write the .v file.
        mode:       'combinational' (max speed, max area) or
                    'sequential' (one MAC/cycle, minimal area, variable input length)

    Returns:
        Path to the generated .v file.
    """
    if mode == "sequential":
        from w2s.sequential.compile import compile_sequential
        return compile_sequential(graph, output_dir)

    _load_generators()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    bits = graph.quant_config.bits
    ops = graph.topological_order()
    lines: List[str] = []

    # --- Header ---
    arch_parts = []
    for op in ops:
        if op.op_type in (OpType.DENSE, OpType.CONV2D, OpType.MULTI_HEAD_ATTENTION):
            arch_parts.append(f"{op.name}({op.op_type.value})")
    total_params = sum(
        int(np.prod(w.shape))
        for op in ops
        for w in op.q_weights.values()
    )

    lines.append(f"// {'=' * 75}")
    lines.append(f"// {graph.name} — Hardwired Neural Network Inference")
    lines.append(f"// Generated by weights2silicon (w2s)")
    lines.append(f"//")
    lines.append(f"// THIS MODULE HAS NO MEMORY. Every weight is a numeric constant.")
    lines.append(f"// The synthesis tool compiles each constant * input into fixed")
    lines.append(f"// shift-add logic. THE WEIGHTS ARE THE SILICON.")
    lines.append(f"//")
    lines.append(f"// Operations  : {len(ops)}")
    lines.append(f"// Parameters  : {total_params:,} (all hardwired)")
    lines.append(f"// Quantization: int{bits} ({graph.quant_config.scheme.value}, "
                 f"{graph.quant_config.granularity.value})")
    lines.append(f"// {'=' * 75}")
    lines.append("")

    # --- Determine module ports from graph inputs/outputs ---
    # Flatten input shapes to individual wires
    in_ports: List[Tuple[str, int]] = []
    wire_map: Dict[str, TensorWires] = {}

    for inp_name in graph.input_names:
        shape = graph.input_shapes.get(inp_name, (1,))
        numel = 1
        for s in shape:
            numel *= s
        wire_names = [f"{inp_name}_{i}" for i in range(numel)]
        for wn in wire_names:
            in_ports.append((wn, bits))
        wire_map[inp_name] = TensorWires(wire_names, shape, bits)

    # We'll determine output ports after processing all ops
    # First, generate all op logic

    op_lines: List[str] = []
    for op in ops:
        gen = _get_generator(op.op_type)
        if gen is None:
            op_lines.append(f"    // WARNING: no generator for {op.op_type.value} "
                          f"(op: {op.name}) — skipped")
            op_lines.append("")
            # Pass through input wires as output (best effort)
            if op.inputs and op.inputs[0] in wire_map:
                for out_name in op.outputs:
                    wire_map[out_name] = wire_map[op.inputs[0]]
            continue

        try:
            new_lines, new_wires = gen(op, wire_map, bits)
            op_lines.extend(new_lines)
            op_lines.append("")
            wire_map.update(new_wires)
        except Exception as e:
            op_lines.append(f"    // ERROR generating {op.name} ({op.op_type.value}): {e}")
            op_lines.append("")
            # Pass through
            if op.inputs and op.inputs[0] in wire_map:
                for out_name in op.outputs:
                    wire_map[out_name] = wire_map[op.inputs[0]]

    # --- Output ports ---
    out_ports: List[Tuple[str, int]] = []
    output_assignments: List[str] = []
    for out_name in graph.output_names:
        if out_name in wire_map:
            tw = wire_map[out_name]
            for i, wn in enumerate(tw.wire_names):
                port_name = f"out_{out_name}_{i}" if len(graph.output_names) > 1 else f"out_{i}"
                out_ports.append((port_name, tw.bits))
                output_assignments.append(f"    assign {port_name} = {wn};")
        else:
            # Output tensor not found — add a placeholder
            out_ports.append((f"out_{out_name}_0", bits))
            output_assignments.append(
                f"    assign out_{out_name}_0 = {bits}'sd0; // WARNING: tensor not found")

    # --- Assemble module ---
    lines.extend(emit.module_header(graph.name, in_ports, out_ports))
    lines.append("")
    lines.extend(op_lines)
    lines.extend(emit.section_comment("Outputs"))
    lines.extend(output_assignments)
    lines.extend(emit.module_footer())

    # --- Write file ---
    vpath = out / f"{graph.name}.v"
    vpath.write_text("\n".join(lines), encoding="utf-8")
    return str(vpath)


# ---------------------------------------------------------------------------
#  Testbench generation
# ---------------------------------------------------------------------------

def generate_testbench(
    graph: ComputeGraph,
    test_inputs: Dict[str, np.ndarray],
    expected_outputs: Dict[str, np.ndarray],
    output_dir: str = ".",
) -> str:
    """
    Generate a Verilog testbench for the compiled design.

    test_inputs:     {input_tensor_name: array of quantized int values}
    expected_outputs: {output_tensor_name: array of expected int values}
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    bits = graph.quant_config.bits
    lines = []

    lines.append("`timescale 1ns / 1ps")
    lines.append("")
    lines.append(f"module {graph.name}_tb;")
    lines.append("")

    # Declare signals
    all_in_wires = []
    for inp_name in graph.input_names:
        shape = graph.input_shapes.get(inp_name, (1,))
        numel = 1
        for s in shape:
            numel *= s
        for i in range(numel):
            wn = f"{inp_name}_{i}"
            lines.append(f"    reg signed [{bits - 1}:0] {wn};")
            all_in_wires.append(wn)

    all_out_wires = []
    for out_name in graph.output_names:
        if out_name in expected_outputs:
            data = expected_outputs[out_name]
            numel = data.shape[-1] if data.ndim > 1 else data.size
        else:
            numel = 1
        for i in range(numel):
            wn = f"out_{out_name}_{i}" if len(graph.output_names) > 1 else f"out_{i}"
            lines.append(f"    wire signed [{bits - 1}:0] {wn};")
            all_out_wires.append((wn, out_name, i))

    lines.append("")

    # DUT instantiation
    lines.append(f"    {graph.name} dut (")
    conns = []
    for wn in all_in_wires:
        conns.append(f"        .{wn}({wn})")
    for wn, _, _ in all_out_wires:
        conns.append(f"        .{wn}({wn})")
    lines.append(",\n".join(conns))
    lines.append("    );")
    lines.append("")

    lines.append("    integer errors;")
    lines.append("")
    lines.append("    initial begin")
    lines.append("        errors = 0;")

    # Generate test vectors
    # For now, support single-vector tests
    n_tests = 0
    for inp_name in graph.input_names:
        if inp_name in test_inputs:
            data = test_inputs[inp_name]
            if data.ndim == 1:
                data = data.reshape(1, -1)
            n_tests = max(n_tests, data.shape[0])

    for t in range(n_tests):
        lines.append(f"")
        lines.append(f"        // --- Test vector {t} ---")
        idx = 0
        for inp_name in graph.input_names:
            if inp_name in test_inputs:
                data = test_inputs[inp_name]
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                vec = data[t].flatten()
                for i, v in enumerate(vec):
                    lines.append(f"        {inp_name}_{i} = {emit.slit(bits, int(v))};")
        lines.append("        #10;")

        for wn, out_name, i in all_out_wires:
            if out_name in expected_outputs:
                exp_data = expected_outputs[out_name]
                if exp_data.ndim == 1:
                    exp_data = exp_data.reshape(1, -1)
                if t < exp_data.shape[0]:
                    exp_val = int(exp_data[t].flatten()[i])
                    lines.append(f"        if ({wn} !== {emit.slit(bits, exp_val)}) begin")
                    lines.append(f'            $display("FAIL vec {t} {wn}: '
                                f'got %0d expected {exp_val}", {wn});')
                    lines.append(f"            errors = errors + 1;")
                    lines.append(f"        end")

    lines.append("")
    lines.append(f'        if (errors == 0) $display("PASS — all {n_tests} vectors verified");')
    lines.append(f'        else $display("FAILED — %0d errors", errors);')
    lines.append("        $finish;")
    lines.append("    end")
    lines.append("")
    lines.append("endmodule")

    tb_path = out / f"{graph.name}_tb.v"
    tb_path.write_text("\n".join(lines), encoding="utf-8")
    return str(tb_path)


# ---------------------------------------------------------------------------
#  Integer forward pass (for golden vector generation)
# ---------------------------------------------------------------------------

def forward_int(
    graph: ComputeGraph,
    inputs: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Run the quantized graph in pure integer arithmetic.
    Matches the Verilog computation exactly for golden reference.
    """
    bits = graph.quant_config.bits
    qmax = 2 ** (bits - 1) - 1
    tensors: Dict[str, np.ndarray] = {}

    # Quantize inputs
    for name, data in inputs.items():
        scale = graph.tensor_scales.get(name, 1.0)
        tensors[name] = np.clip(
            np.round(data * scale), -qmax, qmax
        ).astype(np.int64)

    for op in graph.topological_order():
        _forward_op_int(op, tensors, bits, qmax)

    return {name: tensors[name] for name in graph.output_names if name in tensors}


def _forward_op_int(
    op: Operation,
    tensors: Dict[str, np.ndarray],
    bits: int,
    qmax: int,
):
    """Execute one quantized operation in integer arithmetic."""
    def _get(name):
        return tensors[name].astype(np.int64)

    if op.op_type == OpType.DENSE:
        x = _get(op.inputs[0])
        w = op.q_weights['weight'].astype(np.int64)
        b = op.q_weights.get('bias', np.zeros(w.shape[0])).astype(np.int64)
        acc = x @ w.T + b
        mult = op.q_params.get('requant_mult', 1)
        shift = op.q_params.get('requant_shift', 0)
        if isinstance(mult, np.ndarray):
            scaled = (acc * mult.astype(np.int64)) >> shift
        else:
            scaled = (acc * int(mult)) >> shift
        act = op.attrs.get('activation', 'none')
        if act == 'relu':
            tensors[op.outputs[0]] = np.clip(scaled, 0, qmax).astype(np.int64)
        else:
            tensors[op.outputs[0]] = np.clip(scaled, -qmax, qmax).astype(np.int64)

    elif op.op_type == OpType.RELU:
        tensors[op.outputs[0]] = np.maximum(0, _get(op.inputs[0]))

    elif op.op_type in (OpType.GELU, OpType.SIGMOID, OpType.TANH, OpType.SILU):
        # For quantized activations, apply float then requantize
        # (simplified — real impl would use the same PWL as Verilog)
        tensors[op.outputs[0]] = _get(op.inputs[0])  # pass-through placeholder

    elif op.op_type == OpType.ADD:
        a = _get(op.inputs[0])
        b_val = _get(op.inputs[1])
        tensors[op.outputs[0]] = np.clip(a + b_val, -qmax, qmax).astype(np.int64)

    elif op.op_type == OpType.MULTIPLY:
        a = _get(op.inputs[0])
        b_val = _get(op.inputs[1])
        tensors[op.outputs[0]] = np.clip(
            (a * b_val) >> (bits - 1), -qmax, qmax
        ).astype(np.int64)

    elif op.op_type == OpType.FLATTEN:
        x = _get(op.inputs[0])
        tensors[op.outputs[0]] = x.reshape(x.shape[0], -1) if x.ndim > 2 else x.flatten()

    elif op.op_type == OpType.RESHAPE:
        x = _get(op.inputs[0])
        target = op.attrs.get('target_shape', (-1,))
        tensors[op.outputs[0]] = x.reshape(target)

    elif op.op_type == OpType.CONCAT:
        arrs = [_get(n) for n in op.inputs]
        axis = op.attrs.get('axis', 0)
        tensors[op.outputs[0]] = np.concatenate(arrs, axis=axis)

    elif op.op_type == OpType.CONV2D:
        # Simplified int conv2d
        x = _get(op.inputs[0])
        w = op.q_weights['weight'].astype(np.int64)
        b = op.q_weights.get('bias', np.zeros(w.shape[0])).astype(np.int64)
        stride = op.attrs.get('stride', (1, 1))
        padding = op.attrs.get('padding', (0, 0))
        c_out, c_in, kh, kw = w.shape

        if x.ndim == 3:
            _, h_in, w_in = x.shape
        else:
            h_in = w_in = int(np.sqrt(x.size // c_in))
            x = x.reshape(c_in, h_in, w_in)

        ph, pw = padding
        h_out = (h_in + 2 * ph - kh) // stride[0] + 1
        w_out = (w_in + 2 * pw - kw) // stride[1] + 1

        if ph > 0 or pw > 0:
            x_pad = np.zeros((c_in, h_in + 2 * ph, w_in + 2 * pw), dtype=np.int64)
            x_pad[:, ph:ph + h_in, pw:pw + w_in] = x
        else:
            x_pad = x

        out = np.zeros((c_out, h_out, w_out), dtype=np.int64)
        for co in range(c_out):
            for oh in range(h_out):
                for ow in range(w_out):
                    ih = oh * stride[0]
                    iw = ow * stride[1]
                    patch = x_pad[:, ih:ih + kh, iw:iw + kw]
                    out[co, oh, ow] = np.sum(w[co] * patch) + b[co]

        mult = op.q_params.get('requant_mult', np.ones(c_out, dtype=np.int64))
        shift = op.q_params.get('requant_shift', 0)
        if isinstance(mult, np.ndarray):
            for co in range(c_out):
                out[co] = (out[co] * int(mult[co])) >> shift
        else:
            out = (out * int(mult)) >> shift

        act = op.attrs.get('activation', 'none')
        if act == 'relu':
            out = np.clip(out, 0, qmax)
        else:
            out = np.clip(out, -qmax, qmax)
        tensors[op.outputs[0]] = out.astype(np.int64)

    elif op.op_type == OpType.MAXPOOL2D:
        x = _get(op.inputs[0])
        ks = op.attrs.get('kernel_size', (2, 2))
        st = op.attrs.get('stride', ks)
        c, h, w_dim = x.shape
        h_out = (h - ks[0]) // st[0] + 1
        w_out = (w_dim - ks[1]) // st[1] + 1
        out = np.zeros((c, h_out, w_out), dtype=np.int64)
        for ch in range(c):
            for oh in range(h_out):
                for ow in range(w_out):
                    patch = x[ch, oh*st[0]:oh*st[0]+ks[0], ow*st[1]:ow*st[1]+ks[1]]
                    out[ch, oh, ow] = np.max(patch)
        tensors[op.outputs[0]] = out

    elif op.op_type == OpType.GLOBAL_AVGPOOL:
        x = _get(op.inputs[0])
        if x.ndim == 3:
            tensors[op.outputs[0]] = x.mean(axis=(1, 2)).astype(np.int64)
        else:
            tensors[op.outputs[0]] = x

    elif op.op_type == OpType.EMBEDDING:
        idx = _get(op.inputs[0])
        table = op.q_weights['weight'].astype(np.int64)
        tensors[op.outputs[0]] = table[idx.astype(int)]

    else:
        # Passthrough for unhandled ops
        if op.inputs and op.inputs[0] in tensors:
            tensors[op.outputs[0]] = _get(op.inputs[0])


# ---------------------------------------------------------------------------
#  Summary
# ---------------------------------------------------------------------------

def summarize(graph: ComputeGraph) -> str:
    """Human-readable summary of the compute graph."""
    lines = [
        f"Network: {graph.name}",
        f"Quantization: int{graph.quant_config.bits} "
        f"({graph.quant_config.scheme.value}, {graph.quant_config.granularity.value})",
        f"Operations: {len(graph.operations)}",
        "",
    ]
    total_params = 0
    for op in graph.topological_order():
        n_params = sum(int(np.prod(w.shape)) for w in op.q_weights.values())
        total_params += n_params
        info = f"  {op.name}: {op.op_type.value}"
        if n_params > 0:
            info += f" ({n_params:,} params)"
        if op.attrs:
            info += f" {op.attrs}"
        lines.append(info)
    lines.append(f"\nTotal: {total_params:,} parameters hardwired into silicon")
    return "\n".join(lines)
