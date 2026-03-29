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
from w2s import (
    DenseLayer, quantize, forward_int, forward_float,
    generate_verilog, generate_testbench, summarize,
)


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

    # Define the network for w2s
    # Note: for the hardwired chip, we use the pre-sigmoid linear output.
    # Positive output -> 1, non-positive -> 0.
    layers = [
        DenseLayer(weights=W1, biases=b1, activation="relu"),
        DenseLayer(weights=W2, biases=b2, activation="none"),
    ]

    # Float forward pass for reference
    float_out = forward_float(layers, X)
    print("Float outputs (pre-sigmoid):")
    for i, (inp, out) in enumerate(zip(X, float_out)):
        label = int(out[0] > 0)
        print(f"  {inp} -> {out[0]:+.4f}  (class: {label})")
    print()

    # Quantize
    qnet = quantize("xor_nn", layers, calibration_data=X, bits=8)
    print(summarize(qnet))
    print()

    # Integer forward pass (this is what the Verilog computes)
    int_out = forward_int(qnet, X)
    if int_out.ndim == 1:
        int_out = int_out.reshape(-1, 1)
    print("Quantized integer outputs (matches Verilog exactly):")
    for i, (inp, out) in enumerate(zip(X, int_out)):
        label = int(out[0] > 0)
        target = int(y[i][0])
        status = "OK" if label == target else "WRONG"
        print(f"  {inp} -> {out[0]:+4d}  (class: {label})  [{status}]")
    print()

    # Check correctness
    int_preds = (int_out[:, 0] > 0).astype(int)
    target_preds = y.flatten().astype(int)
    if np.array_equal(int_preds, target_preds):
        print("Quantized model classifies all XOR inputs correctly!")
    else:
        print("WARNING: Quantization degraded accuracy (try more training epochs)")
    print()

    # Generate Verilog
    output_dir = str(Path(__file__).resolve().parent.parent / "output")
    v_path = generate_verilog(qnet, output_dir)
    tb_path = generate_testbench(qnet, X, output_dir)

    print(f"Generated Verilog   : {v_path}")
    print(f"Generated testbench : {tb_path}")
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
