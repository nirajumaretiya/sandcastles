"""
XOR Demo — Train a tiny neural network to solve XOR, then compile it to silicon.

This is the "hello world" of weights2silicon. A 2->8->1 network learns XOR
using plain numpy, then w2s compiles it into synthesizable Verilog where
every weight is a constant in the logic fabric. No memory, no loading —
the weights ARE the silicon.

Run:
    cd weights2silicon
    python examples/xor_demo.py
"""

import sys
from pathlib import Path

# Add project root to path so we can import w2s
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from w2s.core import QuantConfig
from w2s.importers.builder import GraphBuilder
from w2s.quantize import quantize_graph
from w2s.graph import compile_graph, forward_int, summarize


# ---------------------------------------------------------------------------
#  1. Train a tiny XOR network (pure numpy, no frameworks needed)
# ---------------------------------------------------------------------------

def train_xor():
    """Train a 2 -> 8 -> 1 network to solve XOR with backprop."""
    np.random.seed(42)

    # Data
    X = np.array([[0.0, 0.0],
                   [0.0, 1.0],
                   [1.0, 0.0],
                   [1.0, 1.0]])
    y = np.array([[0.0], [1.0], [1.0], [0.0]])

    # Init weights (He initialization)
    W1 = np.random.randn(8, 2) * np.sqrt(2.0 / 2)
    b1 = np.zeros(8)
    W2 = np.random.randn(1, 8) * np.sqrt(2.0 / 8)
    b2 = np.zeros(1)

    lr = 0.05
    for epoch in range(5000):
        # --- Forward ---
        z1 = X @ W1.T + b1
        a1 = np.maximum(0, z1)          # ReLU
        z2 = a1 @ W2.T + b2
        a2 = 1.0 / (1.0 + np.exp(-z2))  # Sigmoid output

        # --- Loss (binary cross-entropy, but MSE works fine here) ---
        loss = np.mean((a2 - y) ** 2)

        # --- Backward ---
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

    predictions = (a2 > 0.5).astype(int).flatten()
    print(f"Training complete — loss: {loss:.6f}")
    print(f"Float predictions: {predictions.tolist()}  (target: [0, 1, 1, 0])")

    return W1, b1, W2, b2, X, y


# ---------------------------------------------------------------------------
#  2. Compile to silicon
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print(" weights2silicon — XOR Demo")
    print(" Training a 2->8->1 network, then compiling to Verilog")
    print("=" * 60)
    print()

    # Train
    W1, b1, W2, b2, X, y = train_xor()
    print()

    # Build compute graph using the modern API
    gb = GraphBuilder("xor_nn")
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

    # Generate Verilog
    output_dir = str(Path(__file__).resolve().parent.parent / "output")
    v_path = compile_graph(graph, output_dir)
    print(f"Generated Verilog: {v_path}")
    print()

    # Integer forward pass (this is what the Verilog computes)
    print("Quantized integer outputs (matches Verilog exactly):")
    all_ok = True
    for i in range(len(X)):
        result = forward_int(graph, {"x": X[i:i+1]})
        out_name = graph.output_names[0]
        val = result[out_name].flatten()[0] if out_name in result else 0
        label = int(val > 0) if isinstance(val, (int, np.integer)) else 0
        target = int(y[i][0])
        status = "OK" if label == target else "WRONG"
        if label != target:
            all_ok = False
        print(f"  {X[i]} -> {val:+4}  (class: {label})  [{status}]")
    print()

    # Check correctness
    if all_ok:
        print("Quantized model classifies all XOR inputs correctly!")
    else:
        print("WARNING: Quantization degraded accuracy (try more training epochs)")
    print()

    # Show a snippet of the generated Verilog
    verilog = Path(v_path).read_text()
    vlines = verilog.split("\n")
    print("--- Generated Verilog (first 50 lines) ---")
    for line in vlines[:50]:
        print(line)
    if len(vlines) > 50:
        print(f"... ({len(vlines) - 50} more lines)")
    print()

    print("=" * 60)
    print(" Done. The weights are now constants in synthesizable Verilog.")
    print(" Next steps:")
    print("   - Simulate with Icarus Verilog:  iverilog -o tb xor_nn.v xor_nn_tb.v && vvp tb")
    print("   - Synthesize with Yosys:         yosys -p 'read_verilog xor_nn.v; synth; stat'")
    print("   - Submit to Tiny Tapeout:         https://tinytapeout.com")
    print("=" * 60)


if __name__ == "__main__":
    main()
