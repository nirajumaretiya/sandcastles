"""
Shared pytest fixtures for the weights2silicon test suite.
"""

import numpy as np
import pytest

from w2s.core import ComputeGraph, QuantConfig
from w2s.importers.builder import GraphBuilder
from w2s.quantize import quantize_graph


# ---------------------------------------------------------------------------
#  XOR network fixture
# ---------------------------------------------------------------------------

def _train_xor():
    """Train a 2->8->1 XOR network with fixed seed. Returns weights."""
    np.random.seed(42)
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([[0.0], [1.0], [1.0], [0.0]])

    W1 = np.random.randn(8, 2) * np.sqrt(2.0 / 2)
    b1 = np.zeros(8)
    W2 = np.random.randn(1, 8) * np.sqrt(2.0 / 8)
    b2 = np.zeros(1)

    lr = 0.05
    for _ in range(5000):
        z1 = X @ W1.T + b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ W2.T + b2
        a2 = 1.0 / (1.0 + np.exp(-z2))
        m = X.shape[0]
        dz2 = (a2 - y) * a2 * (1 - a2) * 2.0 / m
        dW2 = dz2.T @ a1
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ W2
        dz1 = da1 * (z1 > 0).astype(float)
        dW1 = dz1.T @ X
        db1 = dz1.sum(axis=0)
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

    return W1, b1, W2, b2, X, y


@pytest.fixture(scope="session")
def xor_trained_weights():
    """Returns (W1, b1, W2, b2, X, y) for a trained XOR network."""
    return _train_xor()


@pytest.fixture
def xor_graph(xor_trained_weights):
    """Build a ComputeGraph for the XOR network (float, not quantized)."""
    W1, b1, W2, b2, _X, _y = xor_trained_weights
    gb = GraphBuilder("xor_test")
    inp = gb.input("x", shape=(2,))
    h = gb.dense(inp, W1, b1, activation="relu", name="hidden")
    out = gb.dense(h, W2, b2, name="output")
    gb.output(out)
    return gb.build()


@pytest.fixture
def xor_quantized_graph(xor_graph, xor_trained_weights):
    """Build and quantize the XOR graph at int8."""
    _W1, _b1, _W2, _b2, X, _y = xor_trained_weights
    graph = xor_graph
    graph.quant_config = QuantConfig(bits=8)
    quantize_graph(graph, {"x": X})
    return graph


# ---------------------------------------------------------------------------
#  Small CNN fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def cnn_graph():
    """Build a small CNN graph: 1-channel 8x8 input, Conv2D(1->4,3x3), Dense(4*6*6->2)."""
    np.random.seed(99)
    C_out, C_in, kH, kW = 4, 1, 3, 3
    conv_w = np.random.randn(C_out, C_in, kH, kW).astype(np.float64) * 0.5
    conv_b = np.random.randn(C_out).astype(np.float64) * 0.1

    # After conv on 8x8 with no padding: output is 4 x 6 x 6
    flat_dim = C_out * 6 * 6  # = 144
    fc_w = np.random.randn(2, flat_dim).astype(np.float64) * 0.1
    fc_b = np.random.randn(2).astype(np.float64) * 0.01

    gb = GraphBuilder("cnn_test")
    inp = gb.input("image", shape=(1, 8, 8))
    conv_out = gb.conv2d(inp, weight=conv_w, bias=conv_b,
                         stride=(1, 1), padding=(0, 0),
                         activation="relu", name="conv1")
    flat_out = gb.flatten(conv_out, name="flat")
    fc_out = gb.dense(flat_out, weight=fc_w, bias=fc_b, name="fc1")
    gb.output(fc_out)
    return gb.build()


@pytest.fixture
def cnn_quantized_graph(cnn_graph):
    """Quantize the small CNN graph at int8."""
    np.random.seed(99)
    calib = {"image": np.random.randn(10, 1, 8, 8).astype(np.float64) * 0.5}
    cnn_graph.quant_config = QuantConfig(bits=8)
    quantize_graph(cnn_graph, calib)
    return cnn_graph


# ---------------------------------------------------------------------------
#  Output directory fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def output_dir(tmp_path):
    """Provide a temporary output directory."""
    return str(tmp_path / "verilog_out")
