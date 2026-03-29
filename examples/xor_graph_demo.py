"""
XOR Graph Demo — Same XOR network as before, but using the new
compute graph API. Shows the GraphBuilder -> quantize -> compile pipeline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from w2s.core import QuantConfig
from w2s.importers.builder import GraphBuilder
from w2s.quantize import quantize_graph
from w2s.graph import compile_graph, forward_int, summarize


def train_xor():
    """Train 2->8->1 XOR network."""
    np.random.seed(42)
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([[0.0], [1.0], [1.0], [0.0]])

    W1 = np.random.randn(8, 2) * np.sqrt(2.0 / 2)
    b1 = np.zeros(8)
    W2 = np.random.randn(1, 8) * np.sqrt(2.0 / 8)
    b2 = np.zeros(1)

    lr = 0.05
    for epoch in range(5000):
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

        W1 -= lr * dW1; b1 -= lr * db1
        W2 -= lr * dW2; b2 -= lr * db2

    preds = (a2 > 0.5).astype(int).flatten()
    print(f"Training: loss={np.mean((a2-y)**2):.6f}, preds={preds.tolist()}")
    return W1, b1, W2, b2, X, y


def main():
    print("=" * 60)
    print(" weights2silicon — XOR Graph API Demo")
    print("=" * 60)
    print()

    W1, b1, W2, b2, X, y = train_xor()
    print()

    # Build graph
    gb = GraphBuilder("xor_v2")
    inp = gb.input("x", shape=(2,))
    h = gb.dense(inp, W1, b1, activation="relu", name="hidden")
    out = gb.dense(h, W2, b2, name="output")
    gb.output(out)
    graph = gb.build()

    # Quantize
    graph.quant_config = QuantConfig(bits=8)
    quantize_graph(graph, {"x": X})

    print(summarize(graph))
    print()

    # Compile
    output_dir = str(Path(__file__).resolve().parent.parent / "output")
    v_path = compile_graph(graph, output_dir)
    print(f"Generated: {v_path}")

    # Verify
    print("\nQuantized forward pass:")
    for i in range(4):
        result = forward_int(graph, {"x": X[i:i+1]})
        out_name = graph.output_names[0]
        val = result[out_name].flatten()[0] if out_name in result else "?"
        label = int(val > 0) if isinstance(val, (int, np.integer)) else "?"
        target = int(y[i][0])
        print(f"  {X[i]} -> {val:+4}  (class: {label})  [{'OK' if label == target else 'WRONG'}]")

    print()
    print("Done!")


if __name__ == "__main__":
    main()
