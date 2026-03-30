"""
Tests for w2s.quantize — quantization of ComputeGraphs.
"""

import numpy as np
import pytest

from w2s.core import ComputeGraph, OpType, QuantConfig
from w2s.importers.builder import GraphBuilder
from w2s.quantize import quantize_graph


class TestQuantizeXOR:
    def test_does_not_raise(self, xor_graph, xor_trained_weights):
        """quantize_graph should succeed without error on the XOR graph."""
        _W1, _b1, _W2, _b2, X, _y = xor_trained_weights
        xor_graph.quant_config = QuantConfig(bits=8)
        quantize_graph(xor_graph, {"x": X})

    def test_sets_q_weights(self, xor_quantized_graph):
        """After quantization, weighted ops should have q_weights populated."""
        for op in xor_quantized_graph.operations:
            if op.op_type == OpType.DENSE:
                assert "weight" in op.q_weights, f"{op.name} missing q_weights['weight']"

    def test_sets_q_params(self, xor_quantized_graph):
        """After quantization, Dense ops should have q_params populated."""
        for op in xor_quantized_graph.operations:
            if op.op_type == OpType.DENSE:
                assert "requant_mult" in op.q_params, f"{op.name} missing requant_mult"
                assert "requant_shift" in op.q_params, f"{op.name} missing requant_shift"

    def test_q_weights_are_integer(self, xor_quantized_graph):
        for op in xor_quantized_graph.operations:
            if op.op_type == OpType.DENSE:
                w = op.q_weights["weight"]
                assert np.issubdtype(w.dtype, np.integer), (
                    f"{op.name} q_weights dtype should be integer, got {w.dtype}"
                )


class TestQuantizeBitWidths:
    @pytest.mark.parametrize("bits", [4, 8, 16])
    def test_quantize_at_bit_width(self, bits):
        """quantize_graph should work for int4, int8, and int16."""
        np.random.seed(42)
        W1 = np.random.randn(4, 2).astype(np.float64)
        b1 = np.random.randn(4).astype(np.float64)

        gb = GraphBuilder(f"test_{bits}bit")
        inp = gb.input("x", shape=(2,))
        out = gb.dense(inp, W1, b1, name="fc")
        gb.output(out)
        graph = gb.build()
        graph.quant_config = QuantConfig(bits=bits)

        X = np.random.randn(4, 2).astype(np.float64)
        quantize_graph(graph, {"x": X})

        op = graph.get_op("fc")
        assert "weight" in op.q_weights

    @pytest.mark.parametrize("bits", [4, 8, 16])
    def test_quantized_weights_in_range(self, bits):
        """Quantized weights should be within [-qmax, qmax] for the bit width."""
        np.random.seed(42)
        W1 = np.random.randn(4, 2).astype(np.float64)
        b1 = np.random.randn(4).astype(np.float64)

        gb = GraphBuilder("range_test")
        inp = gb.input("x", shape=(2,))
        out = gb.dense(inp, W1, b1, name="fc")
        gb.output(out)
        graph = gb.build()
        graph.quant_config = QuantConfig(bits=bits)

        X = np.random.randn(4, 2).astype(np.float64)
        quantize_graph(graph, {"x": X})

        qmax = 2 ** (bits - 1) - 1
        op = graph.get_op("fc")
        w = op.q_weights["weight"]
        assert np.all(w >= -qmax), f"min q_weight {w.min()} < -{qmax}"
        assert np.all(w <= qmax), f"max q_weight {w.max()} > {qmax}"


class TestQuantizeCNN:
    def test_cnn_quantize(self, cnn_quantized_graph):
        """CNN graph should quantize without error and have q_weights."""
        for op in cnn_quantized_graph.operations:
            if op.op_type in (OpType.DENSE, OpType.CONV2D):
                assert "weight" in op.q_weights, f"{op.name} missing q_weights"
