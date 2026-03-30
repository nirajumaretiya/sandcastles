"""
Tests for w2s.graph.compile_graph — Verilog compilation.
"""

import re
from pathlib import Path

import numpy as np
import pytest

from w2s.core import QuantConfig
from w2s.graph import compile_graph
from w2s.importers.builder import GraphBuilder
from w2s.quantize import quantize_graph


class TestCompileXOR:
    def test_produces_verilog_file(self, xor_quantized_graph, output_dir):
        """compile_graph should produce a .v file."""
        v_path = compile_graph(xor_quantized_graph, output_dir)
        assert Path(v_path).exists()
        assert v_path.endswith(".v")

    def test_verilog_contains_module_name(self, xor_quantized_graph, output_dir):
        """The generated Verilog should contain the graph name as a module."""
        v_path = compile_graph(xor_quantized_graph, output_dir)
        content = Path(v_path).read_text(encoding="utf-8")
        assert "module xor_test" in content

    def test_verilog_contains_endmodule(self, xor_quantized_graph, output_dir):
        """Generated Verilog should have a matching endmodule."""
        v_path = compile_graph(xor_quantized_graph, output_dir)
        content = Path(v_path).read_text(encoding="utf-8")
        assert "endmodule" in content

    def test_verilog_balanced_module(self, xor_quantized_graph, output_dir):
        """Each 'module' keyword should be matched by an 'endmodule'."""
        v_path = compile_graph(xor_quantized_graph, output_dir)
        content = Path(v_path).read_text(encoding="utf-8")
        # Count non-comment module/endmodule tokens
        lines = content.split("\n")
        module_count = 0
        endmodule_count = 0
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("//"):
                continue
            if re.match(r'^module\b', stripped):
                module_count += 1
            if re.match(r'^endmodule\b', stripped):
                endmodule_count += 1
        assert module_count == endmodule_count, (
            f"module count ({module_count}) != endmodule count ({endmodule_count})"
        )


class TestCompileCNN:
    def test_produces_verilog_file(self, cnn_quantized_graph, output_dir):
        """CNN compilation should produce a .v file."""
        v_path = compile_graph(cnn_quantized_graph, output_dir)
        assert Path(v_path).exists()
        assert v_path.endswith(".v")

    def test_verilog_contains_module_name(self, cnn_quantized_graph, output_dir):
        v_path = compile_graph(cnn_quantized_graph, output_dir)
        content = Path(v_path).read_text(encoding="utf-8")
        assert "module cnn_test" in content


class TestCompileSequential:
    def test_sequential_produces_verilog(self, output_dir):
        """Sequential mode compilation should produce a .v file."""
        np.random.seed(42)
        W1 = np.random.randn(4, 2).astype(np.float64) * 0.5
        b1 = np.random.randn(4).astype(np.float64) * 0.1
        W2 = np.random.randn(1, 4).astype(np.float64) * 0.5
        b2 = np.random.randn(1).astype(np.float64) * 0.1

        gb = GraphBuilder("seq_test")
        inp = gb.input("x", shape=(2,))
        h = gb.dense(inp, W1, b1, activation="relu", name="hidden")
        out = gb.dense(h, W2, b2, name="output")
        gb.output(out)
        graph = gb.build()
        graph.quant_config = QuantConfig(bits=8)

        X = np.random.randn(4, 2).astype(np.float64)
        quantize_graph(graph, {"x": X})

        v_path = compile_graph(graph, output_dir, mode="sequential")
        assert Path(v_path).exists()
        assert v_path.endswith(".v")

    def test_sequential_contains_readmemh_or_case(self, output_dir):
        """Sequential Verilog should contain weight ROM (either $readmemh or case)."""
        np.random.seed(42)
        W1 = np.random.randn(4, 2).astype(np.float64) * 0.5
        b1 = np.random.randn(4).astype(np.float64) * 0.1
        W2 = np.random.randn(1, 4).astype(np.float64) * 0.5
        b2 = np.random.randn(1).astype(np.float64) * 0.1

        gb = GraphBuilder("seq_rom")
        inp = gb.input("x", shape=(2,))
        h = gb.dense(inp, W1, b1, activation="relu", name="hidden")
        out = gb.dense(h, W2, b2, name="output")
        gb.output(out)
        graph = gb.build()
        graph.quant_config = QuantConfig(bits=8)

        X = np.random.randn(4, 2).astype(np.float64)
        quantize_graph(graph, {"x": X})

        v_path = compile_graph(graph, output_dir, mode="sequential")
        content = Path(v_path).read_text(encoding="utf-8")
        has_readmemh = "$readmemh" in content
        has_case = "case" in content
        assert has_readmemh or has_case, (
            "Sequential Verilog should contain $readmemh or case for weight ROM"
        )

    def test_sequential_has_clk_and_rst(self, output_dir):
        """Sequential design should declare clk and rst_n ports."""
        np.random.seed(42)
        W1 = np.random.randn(4, 2).astype(np.float64) * 0.5
        b1 = np.random.randn(4).astype(np.float64) * 0.1
        W2 = np.random.randn(1, 4).astype(np.float64) * 0.5
        b2 = np.random.randn(1).astype(np.float64) * 0.1

        gb = GraphBuilder("seq_clk")
        inp = gb.input("x", shape=(2,))
        h = gb.dense(inp, W1, b1, activation="relu", name="hidden")
        out = gb.dense(h, W2, b2, name="output")
        gb.output(out)
        graph = gb.build()
        graph.quant_config = QuantConfig(bits=8)

        X = np.random.randn(4, 2).astype(np.float64)
        quantize_graph(graph, {"x": X})

        v_path = compile_graph(graph, output_dir, mode="sequential")
        content = Path(v_path).read_text(encoding="utf-8")
        assert "clk" in content
        assert "rst_n" in content
