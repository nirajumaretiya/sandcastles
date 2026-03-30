"""Tests for enhanced testbench generation."""

import numpy as np
import pytest

from w2s.graph import generate_testbench, generate_sequential_testbench, forward_int


class TestTestbenchGeneration:
    def test_generates_testbench(self, xor_quantized_graph, xor_trained_weights, output_dir):
        _W1, _b1, _W2, _b2, X, _y = xor_trained_weights
        graph = xor_quantized_graph

        test_inputs = {"x": X[:2]}
        outputs = forward_int(graph, {"x": X[:2]})
        tb_path = generate_testbench(
            graph, test_inputs, outputs, output_dir=output_dir)
        assert tb_path.endswith("_tb.v")
        with open(tb_path) as f:
            content = f.read()
        assert "module" in content
        assert "PASS" in content

    def test_vcd_option(self, xor_quantized_graph, xor_trained_weights, output_dir):
        _W1, _b1, _W2, _b2, X, _y = xor_trained_weights
        graph = xor_quantized_graph

        test_inputs = {"x": X[:1]}
        outputs = forward_int(graph, {"x": X[:1]})
        tb_path = generate_testbench(
            graph, test_inputs, outputs, output_dir=output_dir, vcd=True)
        with open(tb_path) as f:
            content = f.read()
        assert "$dumpfile" in content
        assert "$dumpvars" in content

    def test_tolerance_option(self, xor_quantized_graph, xor_trained_weights, output_dir):
        _W1, _b1, _W2, _b2, X, _y = xor_trained_weights
        graph = xor_quantized_graph

        test_inputs = {"x": X[:1]}
        outputs = forward_int(graph, {"x": X[:1]})
        tb_path = generate_testbench(
            graph, test_inputs, outputs, output_dir=output_dir, tolerance=2)
        with open(tb_path) as f:
            content = f.read()
        assert "total_checks" in content

    def test_sequential_testbench(self, xor_quantized_graph, xor_trained_weights, output_dir):
        _W1, _b1, _W2, _b2, X, _y = xor_trained_weights
        graph = xor_quantized_graph

        test_inputs = {"x": X[:2]}
        outputs = forward_int(graph, {"x": X[:2]})
        tb_path = generate_sequential_testbench(
            graph, test_inputs, outputs, output_dir=output_dir, vcd=True)
        assert "_seq_tb.v" in tb_path
        with open(tb_path) as f:
            content = f.read()
        assert "clk" in content
        assert "rst_n" in content
        assert "data_valid" in content
