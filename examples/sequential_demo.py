"""
Sequential Architecture Demo — Same XOR network, but compiled to a
clocked design with ROM weights and a MAC state machine.

The combinational version uses 16 parallel multipliers (one per weight).
The sequential version uses 1 multiplier + a weight ROM.
Same weights, same silicon concept, dramatically less area.

Run:
    cd weights2silicon
    python examples/sequential_demo.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from w2s.core import QuantConfig
from w2s.importers.builder import GraphBuilder
from w2s.quantize import quantize_graph
from w2s.graph import compile_graph, summarize


def train_xor():
    np.random.seed(42)
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([[0.0], [1.0], [1.0], [0.0]])

    W1 = np.random.randn(8, 2) * np.sqrt(2.0 / 2)
    b1 = np.zeros(8)
    W2 = np.random.randn(1, 8) * np.sqrt(2.0 / 8)
    b2 = np.zeros(1)

    lr = 0.05
    for _ in range(5000):
        z1 = X @ W1.T + b1; a1 = np.maximum(0, z1)
        z2 = a1 @ W2.T + b2; a2 = 1.0 / (1.0 + np.exp(-z2))
        m = X.shape[0]
        dz2 = (a2 - y) * a2 * (1 - a2) * 2.0 / m
        dW2 = dz2.T @ a1; db2 = dz2.sum(axis=0)
        da1 = dz2 @ W2; dz1 = da1 * (z1 > 0).astype(float)
        dW1 = dz1.T @ X; db1 = dz1.sum(axis=0)
        W1 -= lr * dW1; b1 -= lr * db1
        W2 -= lr * dW2; b2 -= lr * db2

    return W1, b1, W2, b2, X, y


def main():
    print("=" * 65)
    print(" weights2silicon — Sequential Architecture Demo")
    print(" Same XOR network, two compilation modes")
    print("=" * 65)
    print()

    W1, b1, W2, b2, X, y = train_xor()

    # Build graph
    gb = GraphBuilder("xor")
    inp = gb.input("x", shape=(2,))
    h = gb.dense(inp, W1, b1, activation="relu", name="hidden")
    out = gb.dense(h, W2, b2, name="output")
    gb.output(out)
    graph = gb.build()
    graph.quant_config = QuantConfig(bits=8)
    quantize_graph(graph, {"x": X})

    print(summarize(graph))
    print()

    output_dir = str(Path(__file__).resolve().parent.parent / "output")

    # Compile BOTH modes
    print("--- Combinational mode (max speed, max area) ---")
    v1 = compile_graph(graph, output_dir, mode="combinational")
    v1_lines = Path(v1).read_text().count("\n")
    print(f"  Generated: {v1}")
    print(f"  Lines: {v1_lines}")
    print()

    print("--- Sequential mode (min area, clocked) ---")
    v2 = compile_graph(graph, output_dir, mode="sequential")
    v2_lines = Path(v2).read_text().count("\n")
    print(f"  Generated: {v2}")
    print(f"  Lines: {v2_lines}")
    print()

    # Show comparison
    print("--- Comparison ---")
    print(f"  Combinational: {v1_lines} lines, 16+8=24 multipliers, ~1 cycle")
    print(f"  Sequential:    {v2_lines} lines,  1 multiplier + ROM, ~24 cycles")
    print()

    # Show a snippet of the sequential Verilog
    seq_v = Path(v2).read_text().split("\n")
    print("--- Sequential Verilog (first 60 lines) ---")
    for line in seq_v[:60]:
        print(line)
    if len(seq_v) > 60:
        print(f"... ({len(seq_v) - 60} more lines)")
    print()

    print("=" * 65)
    print(" The weights are STILL the silicon — they're constants in ROM.")
    print(" The compute is now sequential — one MAC per clock cycle.")
    print(" Variable-length input: just feed more/fewer values.")
    print()
    print(" This is how you fit a real model on a Tiny Tapeout tile.")
    print("=" * 65)


if __name__ == "__main__":
    main()
