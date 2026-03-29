"""
weights2silicon (w2s) — Compile neural network weights to hardwired Verilog.

The missing tool in the open-source weights -> GDS-II -> silicon pipeline.
Takes a trained neural network, quantizes the weights, and generates
synthesizable Verilog where every weight is a numeric constant in the logic.
No memory. No loading. The weights ARE the silicon.

When synthesized, each constant-times-input multiplication becomes a fixed
shift-add circuit. The weight values literally determine the transistor
topology. This is the Taalas concept, open-sourced, at any scale.

Usage:
    from w2s import DenseLayer, quantize, generate_verilog, forward_int

    layers = [
        DenseLayer(W1, b1, 'relu'),
        DenseLayer(W2, b2, 'none'),
    ]
    qnet = quantize("my_nn", layers, calibration_data)
    generate_verilog(qnet, "output/")
"""

import numpy as np
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
#  Data structures
# ---------------------------------------------------------------------------

@dataclass
class DenseLayer:
    """A dense (fully-connected) layer with float weights."""
    weights: np.ndarray   # shape (n_out, n_in)
    biases: np.ndarray    # shape (n_out,)
    activation: str       # 'relu' or 'none'


@dataclass
class QuantizedLayer:
    """A quantized dense layer ready for Verilog generation."""
    weights: np.ndarray     # int64 (n_out, n_in)
    biases: np.ndarray      # int64 (n_out,) — pre-scaled to accumulator units
    n_in: int
    n_out: int
    activation: str
    requant_mult: int       # output = (accumulator * mult) >> shift
    requant_shift: int
    acc_bits: int
    # Original float values kept for Verilog comments
    float_weights: np.ndarray
    float_biases: np.ndarray


@dataclass
class QuantizedNetwork:
    """A fully quantized network ready for Verilog generation."""
    name: str
    layers: List[QuantizedLayer]
    input_scale: float
    bits: int


# ---------------------------------------------------------------------------
#  Quantization
# ---------------------------------------------------------------------------

def quantize(
    name: str,
    layers: List[DenseLayer],
    calibration_data: np.ndarray,
    bits: int = 8,
) -> QuantizedNetwork:
    """
    Quantize a float network to fixed-point integers.

    Uses symmetric per-layer quantization. Calibration data (representative
    inputs) determines the activation ranges at each layer so that the
    integer rescaling factors can be computed at compile time.
    """
    qmax = 2 ** (bits - 1) - 1  # 127 for int8

    # --- Input range ---
    inp_max = float(max(np.max(np.abs(calibration_data)), 1e-10))
    inp_scale = qmax / inp_max

    # --- Calibrate: float forward pass to find activation ranges ---
    act_maxes: List[float] = []
    x = calibration_data.astype(np.float64)
    for layer in layers:
        x = x @ layer.weights.T + layer.biases
        if layer.activation == "relu":
            x = np.maximum(0, x)
        act_maxes.append(float(max(np.max(np.abs(x)), 1e-10)))

    # --- Quantize each layer ---
    qlayers: List[QuantizedLayer] = []
    prev_scale = inp_scale

    for i, layer in enumerate(layers):
        w = layer.weights.astype(np.float64)
        b = layer.biases.astype(np.float64)

        # Weight quantization (symmetric)
        w_max = float(max(np.max(np.abs(w)), 1e-10))
        w_scale = qmax / w_max
        w_q = np.clip(np.round(w * w_scale), -qmax, qmax).astype(np.int64)

        # Bias in accumulator scale: acc_scale = prev_scale * w_scale
        acc_scale = prev_scale * w_scale
        b_q = np.round(b * acc_scale).astype(np.int64)

        # Output scale from calibration
        out_scale = qmax / act_maxes[i]

        # Requantization multiplier:
        #   output_int8 = (accumulator * mult) >> shift
        # where mult/2^shift approximates out_scale / acc_scale
        ratio = out_scale / acc_scale
        shift = 16
        mult = int(round(ratio * (1 << shift)))
        while abs(mult) > 2 ** 31 - 1:
            shift += 1
            mult = int(round(ratio * (1 << shift)))

        # Accumulator width: product is 2*bits, sum of n_in terms
        n_in = w.shape[1]
        acc_bits = 2 * bits + math.ceil(math.log2(max(n_in, 2))) + 2
        acc_bits = max(acc_bits, 24)

        qlayers.append(QuantizedLayer(
            weights=w_q, biases=b_q,
            n_in=n_in, n_out=w.shape[0],
            activation=layer.activation,
            requant_mult=mult, requant_shift=shift,
            acc_bits=acc_bits,
            float_weights=w, float_biases=b,
        ))
        prev_scale = out_scale

    return QuantizedNetwork(name=name, layers=qlayers,
                            input_scale=inp_scale, bits=bits)


# ---------------------------------------------------------------------------
#  Integer forward pass (matches Verilog exactly)
# ---------------------------------------------------------------------------

def forward_int(qnet: QuantizedNetwork, x_float: np.ndarray) -> np.ndarray:
    """
    Run the quantized network in pure integer arithmetic.

    This function replicates the exact computation that the generated
    Verilog performs, so its outputs can be used as golden reference
    values for testbench verification.
    """
    qmax = 2 ** (qnet.bits - 1) - 1
    x = np.clip(np.round(x_float * qnet.input_scale), -qmax, qmax).astype(np.int64)

    for layer in qnet.layers:
        # MAC — matches the Verilog accumulator
        acc = (x @ layer.weights.T) + layer.biases

        # Requantize — matches (acc * mult) >>> shift in Verilog
        scaled = (acc * layer.requant_mult) >> layer.requant_shift

        # Saturate + activate — matches the Verilog clamp/ReLU
        if layer.activation == "relu":
            x = np.clip(scaled, 0, qmax).astype(np.int64)
        else:
            x = np.clip(scaled, -qmax, qmax).astype(np.int64)

    return x


# ---------------------------------------------------------------------------
#  Float forward pass (for comparison)
# ---------------------------------------------------------------------------

def forward_float(layers: List[DenseLayer], x: np.ndarray) -> np.ndarray:
    """Run the original float network."""
    x = x.astype(np.float64)
    for layer in layers:
        x = x @ layer.weights.T + layer.biases
        if layer.activation == "relu":
            x = np.maximum(0, x)
    return x


# ---------------------------------------------------------------------------
#  Verilog generation
# ---------------------------------------------------------------------------

def _slit(bits: int, val: int) -> str:
    """Signed Verilog integer literal."""
    val = int(val)
    if val < 0:
        return f"-{bits}'sd{-val}"
    return f"{bits}'sd{val}"


def generate_verilog(qnet: QuantizedNetwork, output_dir: str = ".") -> str:
    """
    Generate synthesizable Verilog with all weights as numeric constants.

    Returns the file path of the generated .v file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    bits = qnet.bits
    qmax = 2 ** (bits - 1) - 1
    L = []

    def emit(s=""):
        L.append(s)

    # --- Header ---
    arch = " -> ".join(
        [str(qnet.layers[0].n_in)]
        + [str(la.n_out) for la in qnet.layers]
    )
    total_params = sum(la.n_in * la.n_out + la.n_out for la in qnet.layers)

    emit(f"// {'=' * 75}")
    emit(f"// {qnet.name} — Hardwired Neural Network Inference")
    emit(f"// Generated by weights2silicon (w2s)")
    emit(f"//")
    emit(f"// THIS MODULE HAS NO MEMORY. Every weight is a numeric constant.")
    emit(f"// The synthesis tool compiles each constant * input into fixed")
    emit(f"// shift-add logic. THE WEIGHTS ARE THE SILICON.")
    emit(f"//")
    emit(f"// Architecture : {arch}")
    emit(f"// Parameters   : {total_params:,} (all hardwired)")
    emit(f"// Quantization : symmetric int{bits}")
    emit(f"// {'=' * 75}")
    emit()

    n_in = qnet.layers[0].n_in
    n_out = qnet.layers[-1].n_out

    # --- Module ports ---
    emit(f"module {qnet.name} (")
    ports = []
    for i in range(n_in):
        ports.append(f"    input  wire signed [{bits - 1}:0] in_{i}")
    for i in range(n_out):
        ports.append(f"    output wire signed [{bits - 1}:0] out_{i}")
    emit(",\n".join(ports))
    emit(");")
    emit()

    # --- Layer-by-layer generation ---
    prev_names = [f"in_{i}" for i in range(n_in)]

    for li, la in enumerate(qnet.layers):
        act_tag = f", {la.activation.upper()}" if la.activation != "none" else ""
        emit(f"    // {'=' * 71}")
        emit(f"    // Layer {li}: Dense {la.n_in} -> {la.n_out}{act_tag}")
        emit(f"    // {la.n_in * la.n_out} weights + {la.n_out} biases hardwired")
        emit(f"    // Requantize: (acc * {la.requant_mult}) >>> {la.requant_shift}")
        emit(f"    // {'=' * 71}")
        emit()

        layer_outs = []

        for j in range(la.n_out):
            emit(f"    // --- neuron [{li}][{j}] ---")

            # Sign-extend previous-layer outputs to 32 bits for safe arithmetic
            ext_names = []
            for k in range(la.n_in):
                ename = f"l{li}_ext_{j}_{k}"
                emit(f"    wire signed [31:0] {ename} = "
                     f"{{{{24{{{prev_names[k]}[{bits - 1}]}}}}, {prev_names[k]}}};")
                ext_names.append(ename)

            # Accumulator: sum of (weight_constant * sign_extended_input) + bias
            acc_name = f"l{li}_acc_{j}"
            terms = []  # list of (value_str, comment_str)
            for k in range(la.n_in):
                wval = int(la.weights[j, k])
                if wval == 0:
                    continue
                fval = la.float_weights[j, k]
                terms.append((
                    f"({_slit(32, wval)} * {ext_names[k]})",
                    f"W={fval:+.4f}",
                ))

            bval = int(la.biases[j])
            fbval = la.float_biases[j]
            terms.append((
                _slit(32, bval),
                f"bias={fbval:+.4f}",
            ))

            emit(f"    wire signed [31:0] {acc_name} =")
            for ti, (val, cmt) in enumerate(terms):
                prefix = "  " if ti == 0 else "+ "
                suffix = ";" if ti == len(terms) - 1 else ""
                emit(f"    {prefix}{val}{suffix}  // {cmt}")
            emit()

            # Requantization: 64-bit multiply then arithmetic right shift
            ext64 = f"l{li}_ext64_{j}"
            req_name = f"l{li}_req_{j}"
            sh_name = f"l{li}_sh_{j}"
            emit(f"    wire signed [63:0] {ext64} = "
                 f"{{{{32{{{acc_name}[31]}}}}, {acc_name}}};")
            emit(f"    wire signed [63:0] {req_name} = "
                 f"{ext64} * {_slit(64, la.requant_mult)};")
            emit(f"    wire signed [63:0] {sh_name} = "
                 f"{req_name} >>> {la.requant_shift};")
            emit()

            # Saturate + activate
            out_name = f"l{li}_out_{j}"
            if la.activation == "relu":
                emit(f"    wire signed [{bits - 1}:0] {out_name} =")
                emit(f"        ({sh_name} > 64'sd{qmax}) ? {_slit(bits, qmax)} :")
                emit(f"        ({sh_name} < 64'sd0)   ? {bits}'sd0 :")
                emit(f"        {sh_name}[{bits - 1}:0];")
            else:
                emit(f"    wire signed [{bits - 1}:0] {out_name} =")
                emit(f"        ({sh_name} > 64'sd{qmax})  ? {_slit(bits, qmax)} :")
                emit(f"        ({sh_name} < -64'sd{qmax}) ? {_slit(bits, -qmax)} :")
                emit(f"        {sh_name}[{bits - 1}:0];")
            emit()

            layer_outs.append(out_name)

        prev_names = layer_outs

    # --- Output assignments ---
    emit(f"    // {'=' * 71}")
    emit(f"    // Outputs")
    emit(f"    // {'=' * 71}")
    for i in range(n_out):
        emit(f"    assign out_{i} = {prev_names[i]};")
    emit()
    emit("endmodule")

    vpath = out / f"{qnet.name}.v"
    vpath.write_text("\n".join(L), encoding="utf-8")
    return str(vpath)


# ---------------------------------------------------------------------------
#  Testbench generation
# ---------------------------------------------------------------------------

def generate_testbench(
    qnet: QuantizedNetwork,
    test_inputs: np.ndarray,
    output_dir: str = ".",
) -> str:
    """
    Generate a Verilog testbench that checks the combinational module
    against Python-computed golden outputs.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    bits = qnet.bits
    qmax = 2 ** (bits - 1) - 1
    n_in = qnet.layers[0].n_in
    n_out = qnet.layers[-1].n_out

    # Compute golden outputs
    golden = forward_int(qnet, test_inputs)
    if golden.ndim == 1:
        golden = golden.reshape(-1, 1)

    # Quantize inputs for display
    x_q = np.clip(np.round(test_inputs * qnet.input_scale),
                   -qmax, qmax).astype(int)

    L = []

    def emit(s=""):
        L.append(s)

    emit("`timescale 1ns / 1ps")
    emit()
    emit(f"module {qnet.name}_tb;")
    emit()

    for i in range(n_in):
        emit(f"    reg signed [{bits - 1}:0] in_{i};")
    for i in range(n_out):
        emit(f"    wire signed [{bits - 1}:0] out_{i};")
    emit()

    # DUT
    emit(f"    {qnet.name} dut (")
    conns = []
    for i in range(n_in):
        conns.append(f"        .in_{i}(in_{i})")
    for i in range(n_out):
        conns.append(f"        .out_{i}(out_{i})")
    emit(",\n".join(conns))
    emit("    );")
    emit()

    emit("    integer errors;")
    emit()
    emit("    initial begin")
    emit("        errors = 0;")
    emit()

    for t in range(len(test_inputs)):
        emit(f"        // --- Test vector {t} ---")
        for i in range(n_in):
            emit(f"        in_{i} = {_slit(bits, int(x_q[t, i]))};")
        emit("        #10;")
        for i in range(n_out):
            exp = int(golden[t, i])
            emit(f"        if (out_{i} !== {_slit(bits, exp)}) begin")
            emit(f'            $display("FAIL vec {t} out_{i}: '
                 f'got %0d expected {exp}", out_{i});')
            emit(f"            errors = errors + 1;")
            emit(f"        end")
        emit()

    emit(f'        if (errors == 0) $display("PASS — all '
         f'{len(test_inputs)} vectors verified");')
    emit(f'        else $display("FAILED — %0d errors", errors);')
    emit("        $finish;")
    emit("    end")
    emit()
    emit("endmodule")

    tb_path = out / f"{qnet.name}_tb.v"
    tb_path.write_text("\n".join(L), encoding="utf-8")
    return str(tb_path)


# ---------------------------------------------------------------------------
#  Summary / stats
# ---------------------------------------------------------------------------

def summarize(qnet: QuantizedNetwork) -> str:
    """Return a human-readable summary of the quantized network."""
    lines = [
        f"Network: {qnet.name}",
        f"Quantization: symmetric int{qnet.bits}",
        f"Input scale: {qnet.input_scale:.4f}",
        "",
    ]
    total_w = 0
    total_b = 0
    for i, la in enumerate(qnet.layers):
        nw = la.n_in * la.n_out
        total_w += nw
        total_b += la.n_out
        lines.append(
            f"  Layer {i}: {la.n_in} -> {la.n_out} ({la.activation})  "
            f"| {nw} weights, {la.n_out} biases  "
            f"| requant: *{la.requant_mult} >> {la.requant_shift}"
        )
    lines.append(f"\nTotal: {total_w + total_b:,} parameters hardwired into silicon")
    return "\n".join(lines)
