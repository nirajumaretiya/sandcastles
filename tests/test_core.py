"""
Tests for w2s.core — IR definitions (OpType, Operation, ComputeGraph, QuantConfig).
"""

import numpy as np
import pytest

from w2s.core import (
    ComputeGraph,
    OpType,
    Operation,
    QuantConfig,
    QuantGranularity,
    QuantScheme,
    TensorWires,
)


# ---------------------------------------------------------------------------
#  OpType enum
# ---------------------------------------------------------------------------

class TestOpType:
    def test_has_dense(self):
        assert OpType.DENSE.value == "dense"

    def test_has_conv2d(self):
        assert OpType.CONV2D.value == "conv2d"

    def test_has_relu(self):
        assert OpType.RELU.value == "relu"

    def test_has_embedding(self):
        assert OpType.EMBEDDING.value == "embedding"

    def test_has_multi_head_attention(self):
        assert OpType.MULTI_HEAD_ATTENTION.value == "mha"

    def test_has_maxpool2d(self):
        assert OpType.MAXPOOL2D.value == "maxpool2d"

    def test_has_flatten(self):
        assert OpType.FLATTEN.value == "flatten"

    def test_has_add(self):
        assert OpType.ADD.value == "add"

    def test_enum_members_are_complete(self):
        expected = {
            "DENSE", "CONV1D", "CONV2D",
            "RELU", "GELU", "SIGMOID", "TANH", "SILU", "SOFTMAX",
            "LAYERNORM", "RMSNORM", "BATCHNORM",
            "SWIGLU", "ROPE",
            "MULTI_HEAD_ATTENTION", "GROUPED_QUERY_ATTENTION",
            "EMBEDDING", "KV_CACHE",
            "ADD", "MULTIPLY", "RESHAPE", "FLATTEN", "CONCAT",
            "MAXPOOL2D", "AVGPOOL2D", "GLOBAL_AVGPOOL",
        }
        actual = {m.name for m in OpType}
        assert expected.issubset(actual)


# ---------------------------------------------------------------------------
#  Operation
# ---------------------------------------------------------------------------

class TestOperation:
    def test_creation_minimal(self):
        op = Operation(
            op_type=OpType.RELU,
            name="relu_0",
            inputs=["x"],
            outputs=["relu_0_out"],
        )
        assert op.op_type == OpType.RELU
        assert op.name == "relu_0"
        assert op.inputs == ["x"]
        assert op.outputs == ["relu_0_out"]
        assert op.weights == {}
        assert op.q_weights == {}

    def test_creation_with_weights(self):
        w = np.random.randn(4, 3).astype(np.float64)
        b = np.random.randn(4).astype(np.float64)
        op = Operation(
            op_type=OpType.DENSE,
            name="fc1",
            inputs=["x"],
            outputs=["fc1_out"],
            weights={"weight": w, "bias": b},
        )
        assert "weight" in op.weights
        assert "bias" in op.weights
        assert op.weights["weight"].shape == (4, 3)
        assert op.weights["bias"].shape == (4,)

    def test_attrs_default_empty(self):
        op = Operation(
            op_type=OpType.DENSE,
            name="fc",
            inputs=["x"],
            outputs=["fc_out"],
        )
        assert op.attrs == {}

    def test_attrs_can_hold_activation(self):
        op = Operation(
            op_type=OpType.DENSE,
            name="fc",
            inputs=["x"],
            outputs=["fc_out"],
            attrs={"activation": "relu"},
        )
        assert op.attrs["activation"] == "relu"


# ---------------------------------------------------------------------------
#  ComputeGraph
# ---------------------------------------------------------------------------

class TestComputeGraph:
    def test_construction_empty(self):
        g = ComputeGraph(name="test_net")
        assert g.name == "test_net"
        assert g.operations == []
        assert g.input_names == []
        assert g.output_names == []

    def test_add_op_returns_self(self):
        g = ComputeGraph(name="g")
        op = Operation(OpType.RELU, "r", ["x"], ["r_out"])
        result = g.add(op)
        assert result is g
        assert len(g.operations) == 1

    def test_add_op_chaining(self):
        g = ComputeGraph(name="g")
        op1 = Operation(OpType.RELU, "r1", ["x"], ["r1_out"])
        op2 = Operation(OpType.RELU, "r2", ["r1_out"], ["r2_out"])
        g.add(op1).add(op2)
        assert len(g.operations) == 2

    def test_get_op_by_name(self):
        g = ComputeGraph(name="g")
        op = Operation(OpType.DENSE, "fc1", ["x"], ["fc1_out"])
        g.add(op)
        found = g.get_op("fc1")
        assert found is op
        assert g.get_op("nonexistent") is None

    def test_topological_order_linear(self):
        """A -> B -> C should come out in that order."""
        g = ComputeGraph(name="g")
        g.input_names = ["x"]
        a = Operation(OpType.DENSE, "a", ["x"], ["a_out"],
                      weights={"weight": np.eye(2)})
        b = Operation(OpType.RELU, "b", ["a_out"], ["b_out"])
        c = Operation(OpType.DENSE, "c", ["b_out"], ["c_out"],
                      weights={"weight": np.eye(2)})
        # Add in shuffled order
        g.add(c).add(a).add(b)
        order = g.topological_order()
        names = [op.name for op in order]
        assert names.index("a") < names.index("b") < names.index("c")

    def test_topological_order_on_xor_graph(self, xor_graph):
        order = xor_graph.topological_order()
        assert len(order) == len(xor_graph.operations)
        names = [op.name for op in order]
        assert names.index("hidden") < names.index("output")

    def test_default_quant_config(self):
        g = ComputeGraph(name="g")
        assert g.quant_config.bits == 8


# ---------------------------------------------------------------------------
#  QuantConfig
# ---------------------------------------------------------------------------

class TestQuantConfig:
    def test_defaults(self):
        qc = QuantConfig()
        assert qc.bits == 8
        assert qc.scheme == QuantScheme.SYMMETRIC
        assert qc.granularity == QuantGranularity.PER_TENSOR

    def test_custom_bits(self):
        qc = QuantConfig(bits=4)
        assert qc.bits == 4

    def test_custom_scheme(self):
        qc = QuantConfig(scheme=QuantScheme.ASYMMETRIC)
        assert qc.scheme == QuantScheme.ASYMMETRIC

    def test_custom_granularity(self):
        qc = QuantConfig(granularity=QuantGranularity.PER_CHANNEL)
        assert qc.granularity == QuantGranularity.PER_CHANNEL


# ---------------------------------------------------------------------------
#  TensorWires
# ---------------------------------------------------------------------------

class TestTensorWires:
    def test_numel(self):
        tw = TensorWires(wire_names=["a", "b", "c", "d"], shape=(2, 2), bits=8)
        assert tw.numel == 4

    def test_flat_index(self):
        tw = TensorWires(
            wire_names=["w0", "w1", "w2", "w3", "w4", "w5"],
            shape=(2, 3), bits=8,
        )
        # Row-major: index (1, 2) -> 1*3 + 2 = 5
        assert tw.flat(1, 2) == "w5"
        assert tw.flat(0, 0) == "w0"
