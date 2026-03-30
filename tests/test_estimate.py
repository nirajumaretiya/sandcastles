"""
Tests for w2s.estimate — pre-synthesis area estimation.
"""

import numpy as np
import pytest

from w2s.core import QuantConfig
from w2s.estimate import EstimateReport, estimate
from w2s.importers.builder import GraphBuilder
from w2s.quantize import quantize_graph


def _build_simple_graph():
    """Build and quantize a small 2->4->1 graph for estimation tests."""
    np.random.seed(42)
    W1 = np.random.randn(4, 2).astype(np.float64) * 0.5
    b1 = np.random.randn(4).astype(np.float64) * 0.1
    W2 = np.random.randn(1, 4).astype(np.float64) * 0.5
    b2 = np.random.randn(1).astype(np.float64) * 0.1

    gb = GraphBuilder("est_test")
    inp = gb.input("x", shape=(2,))
    h = gb.dense(inp, W1, b1, activation="relu", name="hidden")
    out = gb.dense(h, W2, b2, name="output")
    gb.output(out)
    graph = gb.build()
    graph.quant_config = QuantConfig(bits=8)

    X = np.random.randn(4, 2).astype(np.float64)
    quantize_graph(graph, {"x": X})
    return graph


class TestEstimateBasic:
    def test_returns_estimate_report(self):
        graph = _build_simple_graph()
        report = estimate(graph, mode="combinational")
        assert isinstance(report, EstimateReport)

    def test_combinational_mode(self):
        graph = _build_simple_graph()
        report = estimate(graph, mode="combinational")
        assert report.mode == "combinational"
        assert report.total_params > 0
        assert report.estimated_luts > 0

    def test_sequential_mode(self):
        graph = _build_simple_graph()
        report = estimate(graph, mode="sequential")
        assert report.mode == "sequential"
        assert report.total_params > 0
        assert report.estimated_luts > 0

    def test_has_total_params(self):
        graph = _build_simple_graph()
        report = estimate(graph, mode="combinational")
        # 2->4 (W1: 4*2=8, b1: 4) + 4->1 (W2: 1*4=4, b2: 1) = 17
        assert report.total_params == 17

    def test_report_str_does_not_raise(self):
        """The __str__ method should produce readable output."""
        graph = _build_simple_graph()
        report = estimate(graph, mode="combinational")
        s = str(report)
        assert "combinational" in s.lower() or "Combinational" in s


class TestEstimateComparison:
    def test_sequential_luts_less_than_combinational(self):
        """For the same network, sequential mode should use fewer LUTs."""
        np.random.seed(42)
        # Build a slightly larger graph to make the difference clear
        W1 = np.random.randn(16, 8).astype(np.float64) * 0.5
        b1 = np.random.randn(16).astype(np.float64) * 0.1
        W2 = np.random.randn(4, 16).astype(np.float64) * 0.5
        b2 = np.random.randn(4).astype(np.float64) * 0.1

        gb = GraphBuilder("est_cmp")
        inp = gb.input("x", shape=(8,))
        h = gb.dense(inp, W1, b1, activation="relu", name="hidden")
        out = gb.dense(h, W2, b2, name="output")
        gb.output(out)
        graph = gb.build()
        graph.quant_config = QuantConfig(bits=8)

        X = np.random.randn(4, 8).astype(np.float64)
        quantize_graph(graph, {"x": X})

        comb = estimate(graph, mode="combinational")
        seq = estimate(graph, mode="sequential")
        assert seq.estimated_luts < comb.estimated_luts, (
            f"Sequential ({seq.estimated_luts}) should be < "
            f"combinational ({comb.estimated_luts})"
        )

    def test_sequential_has_more_cycles(self):
        """Sequential mode should report more cycles per inference."""
        graph = _build_simple_graph()
        comb = estimate(graph, mode="combinational")
        seq = estimate(graph, mode="sequential")
        assert seq.cycles_per_inference > comb.cycles_per_inference


class TestEstimateInvalidMode:
    def test_invalid_mode_raises(self):
        graph = _build_simple_graph()
        with pytest.raises(ValueError, match="Unknown mode"):
            estimate(graph, mode="turbo")
