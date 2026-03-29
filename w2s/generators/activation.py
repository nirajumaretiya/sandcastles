"""
activation.py — Verilog generators for activation functions.

Supported activations:
  - ReLU      (exact: clamp negatives to zero)
  - GELU      (8-segment piecewise-linear approximation)
  - Sigmoid   (8-segment piecewise-linear approximation)
  - Tanh      (8-segment piecewise-linear approximation)
  - SiLU      (8-segment piecewise-linear approximation)
  - Softmax   (combinational: max-tree, exp-LUT, sum-tree, reciprocal-mul)

All activations are element-wise (output shape == input shape) except
softmax, which normalises across the entire flattened tensor.
"""

import math
import numpy as np
from typing import List, Dict, Tuple

from w2s.core import Operation, TensorWires
from w2s.emit import (
    slit,
    ulit,
    wire_signed,
    reg_signed,
    sign_extend_expr,
    pwl_lut_lines,
    section_comment,
)


# =========================================================================
#  Helper: apply a PWL element-wise over every wire in the tensor
# =========================================================================

def _elementwise_pwl(
    op: Operation,
    wire_map: Dict[str, TensorWires],
    bits: int,
    breakpoints: List[int],
    slopes: List[int],
    offsets: List[int],
    tag: str,
) -> Tuple[List[str], Dict[str, TensorWires]]:
    """
    Apply a piecewise-linear activation to every element of the input tensor.

    *breakpoints*, *slopes*, *offsets* are passed straight through to
    ``emit.pwl_lut_lines``.  *tag* is used to build unique Verilog names
    (e.g. ``"gelu"``).
    """
    inp = wire_map[op.inputs[0]]
    prefix = op.name

    out_wires: List[str] = []
    lines: List[str] = []
    lines += section_comment(f"{tag} activation — {prefix}")

    for i in range(inp.numel):
        in_w = inp.wire_names[i]
        out_w = f"{prefix}_out_{i}"
        lut_pfx = f"{prefix}_{tag}_{i}"

        lines += pwl_lut_lines(
            input_wire=in_w,
            output_wire=out_w,
            breakpoints=breakpoints,
            slopes=slopes,
            offsets=offsets,
            input_bits=bits,
            output_bits=bits,
            lut_prefix=lut_pfx,
        )
        out_wires.append(out_w)

    tw = TensorWires(
        wire_names=out_wires,
        shape=inp.shape,
        bits=bits,
        signed=True,
    )
    return lines, {op.outputs[0]: tw}


# =========================================================================
#  ReLU
# =========================================================================

def generate_relu(
    op: Operation,
    wire_map: Dict[str, TensorWires],
    bits: int,
) -> Tuple[List[str], Dict[str, TensorWires]]:
    """ReLU: y = max(x, 0).  Pure combinational — one ternary per element."""
    inp = wire_map[op.inputs[0]]
    prefix = op.name

    lines: List[str] = []
    lines += section_comment(f"ReLU — {prefix}")

    out_wires: List[str] = []
    for i in range(inp.numel):
        in_w = inp.wire_names[i]
        out_w = f"{prefix}_out_{i}"
        lines.append(
            f"    wire signed [{bits - 1}:0] {out_w} = "
            f"({in_w}[{bits - 1}]) ? {bits}'sd0 : {in_w};"
        )
        out_wires.append(out_w)

    tw = TensorWires(
        wire_names=out_wires,
        shape=inp.shape,
        bits=bits,
        signed=True,
    )
    return lines, {op.outputs[0]: tw}


# =========================================================================
#  Sigmoid  (piecewise-linear, 8 segments)
# =========================================================================

def _sigmoid_pwl_params(bits: int) -> Tuple[List[int], List[int], List[int]]:
    """
    Return (breakpoints, slopes, offsets) for a sigmoid approximation.

    Input: signed int with *bits* width, treated as fixed-point in
    roughly [-1.0, +1.0] for an 8-bit type (i.e. scale = 1/128).

    Sigmoid(x) = 1 / (1 + exp(-x)).

    We compute the reference curve at float precision, then fit 8 linear
    segments and quantise the slope/offset into Q8 fixed-point so that:
        y_int = (slope * x_int + offset) >> 8
    produces a value in [0, 127] for 8-bit (representing [0.0, ~1.0]).
    """
    qmin = -(1 << (bits - 1))       # -128 for 8-bit
    qmax = (1 << (bits - 1)) - 1    # 127 for 8-bit
    scale = 1.0 / (1 << (bits - 1)) # maps int -> roughly [-1,1]

    # Breakpoints (7 boundaries -> 8 segments) in quantized int space
    breakpoints = [-96, -64, -32, 0, 32, 64, 96]

    # Evaluate sigmoid at segment boundaries + endpoints
    edges = [qmin] + breakpoints + [qmax]

    # Target sigmoid output scaled to [0, qmax]
    def sig_q(x_int):
        x_f = x_int * scale
        return qmax / (1.0 + math.exp(-x_f))

    slopes: List[int] = []
    offsets: List[int] = []
    for seg in range(len(edges) - 1):
        x0 = edges[seg]
        x1 = edges[seg + 1]
        y0 = sig_q(x0)
        y1 = sig_q(x1)
        # Slope in Q8: slope_q8 = (y1 - y0) / (x1 - x0) * 256
        if x1 != x0:
            m = (y1 - y0) / (x1 - x0) * 256.0
        else:
            m = 0.0
        # Offset in Q8: y_int = (m * x_int + b) >> 8,  b = y0*256 - m*x0
        b = y0 * 256.0 - m * x0
        slopes.append(int(round(m)))
        offsets.append(int(round(b)))

    return breakpoints, slopes, offsets


def generate_sigmoid(
    op: Operation,
    wire_map: Dict[str, TensorWires],
    bits: int,
) -> Tuple[List[str], Dict[str, TensorWires]]:
    """Sigmoid activation — 8-segment piecewise-linear approximation."""
    bp, sl, off = _sigmoid_pwl_params(bits)
    return _elementwise_pwl(op, wire_map, bits, bp, sl, off, "sigmoid")


# =========================================================================
#  Tanh  (piecewise-linear, 8 segments)
# =========================================================================

def _tanh_pwl_params(bits: int) -> Tuple[List[int], List[int], List[int]]:
    """
    Return (breakpoints, slopes, offsets) for a tanh approximation.

    tanh(x) output is in [-1, 1], mapped to [-127, 127] for int8.
    """
    qmin = -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1
    scale = 1.0 / (1 << (bits - 1))

    breakpoints = [-96, -64, -32, 0, 32, 64, 96]
    edges = [qmin] + breakpoints + [qmax]

    def tanh_q(x_int):
        x_f = x_int * scale
        return qmax * math.tanh(x_f)

    slopes: List[int] = []
    offsets: List[int] = []
    for seg in range(len(edges) - 1):
        x0 = edges[seg]
        x1 = edges[seg + 1]
        y0 = tanh_q(x0)
        y1 = tanh_q(x1)
        if x1 != x0:
            m = (y1 - y0) / (x1 - x0) * 256.0
        else:
            m = 0.0
        b = y0 * 256.0 - m * x0
        slopes.append(int(round(m)))
        offsets.append(int(round(b)))

    return breakpoints, slopes, offsets


def generate_tanh(
    op: Operation,
    wire_map: Dict[str, TensorWires],
    bits: int,
) -> Tuple[List[str], Dict[str, TensorWires]]:
    """Tanh activation — 8-segment piecewise-linear approximation."""
    bp, sl, off = _tanh_pwl_params(bits)
    return _elementwise_pwl(op, wire_map, bits, bp, sl, off, "tanh")


# =========================================================================
#  GELU  (piecewise-linear, 10 segments)
# =========================================================================

def _gelu_pwl_params(bits: int) -> Tuple[List[int], List[int], List[int]]:
    """
    Return (breakpoints, slopes, offsets) for a GELU approximation.

    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))

    Output range is approximately [-0.17, +inf) but for |x| in [-1,1]
    (the int8 representable range) it stays within [-0.17, 1.0].
    We map output to the signed int range [-128, 127].
    """
    qmin = -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1
    scale = 1.0 / (1 << (bits - 1))

    # 9 boundaries -> 10 segments, finer granularity in the transition region
    breakpoints = [-112, -80, -48, -16, 0, 16, 48, 80, 112]
    edges = [qmin] + breakpoints + [qmax]

    def gelu_float(x_f):
        c = math.sqrt(2.0 / math.pi)
        return 0.5 * x_f * (1.0 + math.tanh(c * (x_f + 0.044715 * x_f ** 3)))

    def gelu_q(x_int):
        x_f = x_int * scale
        g = gelu_float(x_f)
        # Scale output so that gelu(1.0) ~ qmax
        return g * qmax

    slopes: List[int] = []
    offsets: List[int] = []
    for seg in range(len(edges) - 1):
        x0 = edges[seg]
        x1 = edges[seg + 1]
        y0 = gelu_q(x0)
        y1 = gelu_q(x1)
        if x1 != x0:
            m = (y1 - y0) / (x1 - x0) * 256.0
        else:
            m = 0.0
        b = y0 * 256.0 - m * x0
        slopes.append(int(round(m)))
        offsets.append(int(round(b)))

    return breakpoints, slopes, offsets


def generate_gelu(
    op: Operation,
    wire_map: Dict[str, TensorWires],
    bits: int,
) -> Tuple[List[str], Dict[str, TensorWires]]:
    """GELU activation — 10-segment piecewise-linear approximation."""
    bp, sl, off = _gelu_pwl_params(bits)
    return _elementwise_pwl(op, wire_map, bits, bp, sl, off, "gelu")


# =========================================================================
#  SiLU (Swish)  (piecewise-linear, 8 segments)
# =========================================================================

def _silu_pwl_params(bits: int) -> Tuple[List[int], List[int], List[int]]:
    """
    Return (breakpoints, slopes, offsets) for a SiLU approximation.

    SiLU(x) = x * sigmoid(x).
    For x in [-1,1] the output range is roughly [-0.28, 0.73].
    Scaled to the signed int range.
    """
    qmin = -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1
    scale = 1.0 / (1 << (bits - 1))

    breakpoints = [-96, -64, -32, 0, 32, 64, 96]
    edges = [qmin] + breakpoints + [qmax]

    def silu_q(x_int):
        x_f = x_int * scale
        sig = 1.0 / (1.0 + math.exp(-x_f))
        return x_f * sig * qmax   # scale into int range

    slopes: List[int] = []
    offsets: List[int] = []
    for seg in range(len(edges) - 1):
        x0 = edges[seg]
        x1 = edges[seg + 1]
        y0 = silu_q(x0)
        y1 = silu_q(x1)
        if x1 != x0:
            m = (y1 - y0) / (x1 - x0) * 256.0
        else:
            m = 0.0
        b = y0 * 256.0 - m * x0
        slopes.append(int(round(m)))
        offsets.append(int(round(b)))

    return breakpoints, slopes, offsets


def generate_silu(
    op: Operation,
    wire_map: Dict[str, TensorWires],
    bits: int,
) -> Tuple[List[str], Dict[str, TensorWires]]:
    """SiLU (Swish) activation — 8-segment piecewise-linear approximation."""
    bp, sl, off = _silu_pwl_params(bits)
    return _elementwise_pwl(op, wire_map, bits, bp, sl, off, "silu")


# =========================================================================
#  Softmax  (combinational: max-tree + exp-LUT + sum-tree + recip-mul)
# =========================================================================

def _exp_lut_value(x_int: int) -> int:
    """
    Unsigned 8-bit exp approximation for softmax.

    Input  x_int is a non-positive signed int8 value (result of subtracting
    the max), so x_int is in [-255, 0].
    Output is an unsigned 8-bit value in [0, 255] representing
    exp(x_int / 64) scaled so that exp(0) = 255.

    The divisor 64 controls the effective temperature — chosen so the
    exponential decays to ~zero within the representable int8 range.
    """
    x_f = x_int / 64.0
    y = math.exp(x_f) * 255.0
    return max(0, min(255, int(round(y))))


def _build_exp_lut_case(input_wire: str, output_reg: str,
                        prefix: str, bits: int) -> List[str]:
    """
    Generate a 256-entry case statement mapping int8 -> uint8 for exp.

    Only entries in [-255, 0] are meaningful (post max-subtraction).
    Positive inputs are clamped to exp(0)=255.
    """
    lines: List[str] = []
    lines.append(f"    reg [7:0] {output_reg};")
    lines.append(f"    always @(*) begin")
    lines.append(f"        case ({input_wire})")

    for v in range(0, -256, -1):
        ev = _exp_lut_value(v)
        lines.append(f"            {slit(bits, v)}: {output_reg} = {ulit(8, ev)};")

    lines.append(f"            default: {output_reg} = {ulit(8, 255)};")
    lines.append(f"        endcase")
    lines.append(f"    end")
    return lines


def generate_softmax(
    op: Operation,
    wire_map: Dict[str, TensorWires],
    bits: int,
) -> Tuple[List[str], Dict[str, TensorWires]]:
    """
    Softmax — fully combinational implementation.

    Steps:
      1. Comparator tree to find the maximum element.
      2. Subtract max from each element.
      3. Per-element exp via case-statement LUT (256 entries).
      4. Adder tree to compute the sum of exponentials.
      5. Per-element division: out[i] = exp[i] * 127 / sum.
         Division is done via a precomputed reciprocal LUT: for each
         possible sum value *s* the LUT stores round(127 * 256 / s),
         then  out[i] = (exp[i] * recip[sum]) >> 8.

    Output is signed int (0 to 127), representing probabilities in [0, 1).
    """
    inp = wire_map[op.inputs[0]]
    prefix = op.name
    n = inp.numel

    lines: List[str] = []
    lines += section_comment(f"Softmax — {prefix}  ({n} elements)")

    # ------------------------------------------------------------------
    # 1. Comparator tree — find max
    # ------------------------------------------------------------------
    lines += section_comment("Step 1: max-finding comparator tree")

    # Level 0: copy inputs
    current_level: List[str] = []
    for i in range(n):
        w = f"{prefix}_cmp_0_{i}"
        lines.append(
            f"    wire signed [{bits - 1}:0] {w} = {inp.wire_names[i]};"
        )
        current_level.append(w)

    level = 0
    while len(current_level) > 1:
        next_level: List[str] = []
        level += 1
        for j in range(0, len(current_level), 2):
            out_w = f"{prefix}_cmp_{level}_{j // 2}"
            if j + 1 < len(current_level):
                a = current_level[j]
                b = current_level[j + 1]
                lines.append(
                    f"    wire signed [{bits - 1}:0] {out_w} = "
                    f"({a} > {b}) ? {a} : {b};"
                )
            else:
                # Odd element — pass through
                lines.append(
                    f"    wire signed [{bits - 1}:0] {out_w} = "
                    f"{current_level[j]};"
                )
            next_level.append(out_w)
        current_level = next_level

    max_wire = current_level[0]

    # ------------------------------------------------------------------
    # 2. Subtract max from each element
    # ------------------------------------------------------------------
    lines += section_comment("Step 2: subtract max from each element")

    diff_bits = bits + 1  # one extra bit for subtraction result
    diff_wires: List[str] = []
    for i in range(n):
        dw = f"{prefix}_diff_{i}"
        lines.append(
            f"    wire signed [{diff_bits - 1}:0] {dw} = "
            f"{sign_extend_expr(inp.wire_names[i], bits, diff_bits)} - "
            f"{sign_extend_expr(max_wire, bits, diff_bits)};"
        )
        diff_wires.append(dw)

    # ------------------------------------------------------------------
    # 3. Per-element exp LUT  (case statement, shared logic per element)
    # ------------------------------------------------------------------
    lines += section_comment("Step 3: per-element exp LUT")

    exp_wires: List[str] = []
    for i in range(n):
        exp_reg = f"{prefix}_exp_{i}"
        lines += _build_exp_lut_case(
            input_wire=diff_wires[i],
            output_reg=exp_reg,
            prefix=f"{prefix}_elut_{i}",
            bits=diff_bits,
        )
        exp_wires.append(exp_reg)

    # ------------------------------------------------------------------
    # 4. Adder tree for sum of exponentials
    # ------------------------------------------------------------------
    lines += section_comment("Step 4: adder tree for sum of exp values")

    # Width of the sum needs enough bits: up to n * 255
    sum_bits = 8 + math.ceil(math.log2(max(n, 2))) + 1

    # Level 0: extend each exp to sum_bits (unsigned)
    current_sum: List[str] = []
    for i in range(n):
        sw = f"{prefix}_sext_{i}"
        pad = sum_bits - 8
        lines.append(
            f"    wire [{sum_bits - 1}:0] {sw} = "
            f"{{{{{pad}'b0}}, {exp_wires[i]}}};"
        )
        current_sum.append(sw)

    level = 0
    while len(current_sum) > 1:
        next_sum: List[str] = []
        level += 1
        for j in range(0, len(current_sum), 2):
            sw = f"{prefix}_sum_{level}_{j // 2}"
            if j + 1 < len(current_sum):
                lines.append(
                    f"    wire [{sum_bits - 1}:0] {sw} = "
                    f"{current_sum[j]} + {current_sum[j + 1]};"
                )
            else:
                lines.append(
                    f"    wire [{sum_bits - 1}:0] {sw} = "
                    f"{current_sum[j]};"
                )
            next_sum.append(sw)
        current_sum = next_sum

    sum_wire = current_sum[0]

    # ------------------------------------------------------------------
    # 5. Reciprocal LUT + per-element multiply for division
    # ------------------------------------------------------------------
    lines += section_comment("Step 5: reciprocal multiply for division")

    # Reciprocal LUT: for sum value s, store round(127 * 256 / s).
    # The result is in Q8 fixed point so that:
    #     out[i] = (exp[i] * recip) >> 8
    # gives a value in [0, 127].
    #
    # Max possible sum is n * 255.  We build a case statement over
    # meaningful sum values (1 .. n*255).  Sum == 0 should not happen
    # in practice (at least one exp >= 1), mapped to 0.
    max_sum = n * 255
    recip_bits = 16  # enough for recip values up to 127*256 = 32512

    recip_reg = f"{prefix}_recip"
    lines.append(f"    reg [{recip_bits - 1}:0] {recip_reg};")
    lines.append(f"    always @(*) begin")
    lines.append(f"        case ({sum_wire})")

    for s in range(1, max_sum + 1):
        rv = int(round(127.0 * 256.0 / s))
        rv = min(rv, (1 << recip_bits) - 1)
        lines.append(f"            {ulit(sum_bits, s)}: {recip_reg} = {ulit(recip_bits, rv)};")

    lines.append(f"            default: {recip_reg} = {ulit(recip_bits, 0)};")
    lines.append(f"        endcase")
    lines.append(f"    end")

    # Per-element multiply + shift -> signed output [0, 127]
    mul_bits = 8 + recip_bits  # product width

    out_wires: List[str] = []
    for i in range(n):
        prod_w = f"{prefix}_prod_{i}"
        shifted_w = f"{prefix}_psh_{i}"
        out_w = f"{prefix}_out_{i}"

        lines.append(
            f"    wire [{mul_bits - 1}:0] {prod_w} = "
            f"{exp_wires[i]} * {recip_reg};"
        )
        lines.append(
            f"    wire [{mul_bits - 1}:0] {shifted_w} = {prod_w} >> 8;"
        )
        # Saturate to signed [0, 127]
        lines.append(
            f"    wire signed [{bits - 1}:0] {out_w} = "
            f"({shifted_w} > {ulit(mul_bits, 127)}) ? {slit(bits, 127)} : "
            f"{shifted_w}[{bits - 1}:0];"
        )
        out_wires.append(out_w)

    tw = TensorWires(
        wire_names=out_wires,
        shape=inp.shape,
        bits=bits,
        signed=True,
    )
    return lines, {op.outputs[0]: tw}
