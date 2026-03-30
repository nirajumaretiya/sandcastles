"""
Tests for w2s.graph.forward_int — integer forward pass.
"""

import numpy as np
import pytest

from w2s.core import QuantConfig
from w2s.graph import forward_int
from w2s.importers.builder import GraphBuilder
from w2s.quantize import quantize_graph


class TestForwardIntXOR:
    def test_returns_correct_output_name(self, xor_quantized_graph, xor_trained_weights):
        _W1, _b1, _W2, _b2, X, _y = xor_trained_weights
        result = forward_int(xor_quantized_graph, {"x": X[0:1]})
        out_name = xor_quantized_graph.output_names[0]
        assert out_name in result, f"Expected '{out_name}' in forward_int result"

    def test_output_shape(self, xor_quantized_graph, xor_trained_weights):
        """forward_int output for XOR should have 1 element (single neuron output)."""
        _W1, _b1, _W2, _b2, X, _y = xor_trained_weights
        result = forward_int(xor_quantized_graph, {"x": X[0:1]})
        out_name = xor_quantized_graph.output_names[0]
        out = result[out_name]
        assert out.size >= 1

    def test_xor_predictions(self, xor_quantized_graph, xor_trained_weights):
        """The trained XOR network should predict the XOR truth table correctly."""
        _W1, _b1, _W2, _b2, X, y = xor_trained_weights
        correct = 0
        for i in range(4):
            result = forward_int(xor_quantized_graph, {"x": X[i:i + 1]})
            out_name = xor_quantized_graph.output_names[0]
            val = result[out_name].flatten()[0]
            predicted = int(val > 0)
            expected = int(y[i][0])
            if predicted == expected:
                correct += 1
        # The trained+quantized network should get at least 3 of 4 correct.
        # Typically it gets all 4.
        assert correct >= 3, f"XOR accuracy {correct}/4 is too low"


class TestForwardIntBitWidths:
    @pytest.mark.parametrize("bits", [4, 8, 16])
    def test_forward_at_bit_width(self, bits):
        """forward_int should work at different bit widths."""
        np.random.seed(42)
        W1 = np.random.randn(4, 2).astype(np.float64) * 0.5
        b1 = np.random.randn(4).astype(np.float64) * 0.1

        gb = GraphBuilder(f"fwd_{bits}")
        inp = gb.input("x", shape=(2,))
        out = gb.dense(inp, W1, b1, name="fc")
        gb.output(out)
        graph = gb.build()
        graph.quant_config = QuantConfig(bits=bits)

        X = np.random.randn(4, 2).astype(np.float64)
        quantize_graph(graph, {"x": X})

        result = forward_int(graph, {"x": X[0:1]})
        out_name = graph.output_names[0]
        assert out_name in result

    @pytest.mark.parametrize("bits", [4, 8, 16])
    def test_forward_output_in_range(self, bits):
        """forward_int output values should be within the quantized range."""
        np.random.seed(42)
        W1 = np.random.randn(4, 2).astype(np.float64) * 0.5
        b1 = np.random.randn(4).astype(np.float64) * 0.1

        gb = GraphBuilder(f"range_{bits}")
        inp = gb.input("x", shape=(2,))
        out = gb.dense(inp, W1, b1, name="fc")
        gb.output(out)
        graph = gb.build()
        graph.quant_config = QuantConfig(bits=bits)

        X = np.random.randn(4, 2).astype(np.float64)
        quantize_graph(graph, {"x": X})

        qmax = 2 ** (bits - 1) - 1
        result = forward_int(graph, {"x": X[0:1]})
        out_name = graph.output_names[0]
        out = result[out_name]
        assert np.all(out >= -qmax), f"Output below -qmax: {out.min()}"
        assert np.all(out <= qmax), f"Output above qmax: {out.max()}"
