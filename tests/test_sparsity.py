"""Tests for the sparsity analysis and pruning module."""

import numpy as np
import pytest

from w2s.core import QuantConfig
from w2s.quantize import quantize_graph
from w2s.sparsity import (
    analyze_sparsity,
    prune_weights,
    enforce_structured_2_4,
    detect_structured_2_4,
    detect_structured_nm,
)


class TestAnalyzeSparsity:
    def test_returns_report(self, xor_quantized_graph):
        report = analyze_sparsity(xor_quantized_graph)
        assert report.total_weights > 0

    def test_report_str(self, xor_quantized_graph):
        report = analyze_sparsity(xor_quantized_graph)
        s = str(report)
        assert "Sparsity Report" in s

    def test_layers_populated(self, xor_quantized_graph):
        report = analyze_sparsity(xor_quantized_graph)
        assert len(report.layers) > 0

    def test_sparsity_between_0_and_1(self, xor_quantized_graph):
        report = analyze_sparsity(xor_quantized_graph)
        assert 0.0 <= report.overall_sparsity <= 1.0


class TestPruneWeights:
    def test_prune_increases_sparsity(self, xor_quantized_graph):
        before = analyze_sparsity(xor_quantized_graph)
        prune_weights(xor_quantized_graph, threshold=5)
        after = analyze_sparsity(xor_quantized_graph)
        assert after.overall_sparsity >= before.overall_sparsity

    def test_prune_to_target(self, xor_quantized_graph):
        prune_weights(xor_quantized_graph, target_sparsity=0.5)
        report = analyze_sparsity(xor_quantized_graph)
        # Should be approximately 50% sparse (within tolerance)
        assert report.overall_sparsity >= 0.4


class TestStructured24:
    def test_detect_on_sparse_matrix(self):
        """A matrix with exactly 2 nonzero per group of 4 should be detected."""
        w = np.array([
            [1, 0, 2, 0, 3, 0, 4, 0],
            [0, 1, 0, 2, 0, 3, 0, 4],
        ])
        assert detect_structured_2_4(w) is True

    def test_detect_on_dense_matrix(self):
        """A fully dense matrix should not be detected as 2:4."""
        w = np.ones((4, 8), dtype=np.int64)
        assert detect_structured_2_4(w) is False

    def test_enforce_2_4(self, xor_quantized_graph):
        enforce_structured_2_4(xor_quantized_graph)
        # Check that all weight groups have at most 2 nonzero per 4
        for op in xor_quantized_graph.operations:
            for key in ('weight',):
                if key in op.q_weights:
                    w = op.q_weights[key]
                    if w.ndim >= 2 and w.shape[-1] >= 4:
                        flat = w.reshape(-1, w.shape[-1])
                        n_groups = flat.shape[1] // 4
                        for row in flat:
                            for g in range(n_groups):
                                group = row[g * 4:(g + 1) * 4]
                                assert np.count_nonzero(group) <= 2


class TestDetectNM:
    def test_detect_1_4(self):
        w = np.array([
            [1, 0, 0, 0, 2, 0, 0, 0],
            [0, 1, 0, 0, 0, 2, 0, 0],
        ])
        result = detect_structured_nm(w)
        assert result is not None
        assert result[0] == 1 and result[1] == 4
