"""
emit.py — Verilog code-emission helpers used by all generators.

Every function here produces fragments of synthesizable Verilog-2001.
Generators call these instead of hand-formatting strings, keeping the
output consistent and correct across all operation types.
"""

from typing import List, Tuple
import math


# ---------------------------------------------------------------------------
#  Literals
# ---------------------------------------------------------------------------

def slit(bits: int, val: int) -> str:
    """Signed Verilog literal.  slit(8, -5) -> \"-8'sd5\" """
    val = int(val)
    if val < 0:
        return f"-{bits}'sd{-val}"
    return f"{bits}'sd{val}"


def ulit(bits: int, val: int) -> str:
    """Unsigned Verilog literal."""
    return f"{bits}'d{int(val)}"


# ---------------------------------------------------------------------------
#  Wire helpers
# ---------------------------------------------------------------------------

def wire_signed(name: str, bits: int) -> str:
    """Declare a signed wire: ``wire signed [7:0] foo;``"""
    return f"    wire signed [{bits - 1}:0] {name};"


def reg_signed(name: str, bits: int) -> str:
    """Declare a signed reg: ``reg signed [7:0] foo;``"""
    return f"    reg signed [{bits - 1}:0] {name};"


def sign_extend_expr(wire_name: str, from_bits: int, to_bits: int) -> str:
    """
    Expression that sign-extends *wire_name* from *from_bits* to *to_bits*.

    Example (8 -> 32):
        ``{{24{foo[7]}}, foo}``
    """
    pad = to_bits - from_bits
    return f"{{{{{pad}{{{wire_name}[{from_bits - 1}]}}}}, {wire_name}}}"


def sign_extend_wire(src: str, src_bits: int, dst: str, dst_bits: int) -> str:
    """Full wire declaration + assignment for sign extension."""
    expr = sign_extend_expr(src, src_bits, dst_bits)
    return f"    wire signed [{dst_bits - 1}:0] {dst} = {expr};"


# ---------------------------------------------------------------------------
#  Arithmetic building blocks
# ---------------------------------------------------------------------------

def mac_term(weight_val: int, input_wire: str, acc_bits: int = 32,
             comment: str = "") -> str:
    """
    One multiply-accumulate term: ``(32'sd85 * input_wire)``

    Returns the expression string (no leading +, no trailing ;).
    The caller is responsible for joining terms and terminating.
    """
    cmt = f"  // {comment}" if comment else ""
    return f"({slit(acc_bits, weight_val)} * {input_wire}){cmt}"


def requantize_lines(
    acc_name: str,
    acc_bits: int,
    mult: int,
    shift: int,
    prefix: str,
) -> Tuple[List[str], str]:
    """
    Generate the 64-bit requantization multiply + shift.

    Returns (verilog_lines, shifted_wire_name).
    """
    ext64 = f"{prefix}_ext64"
    prod = f"{prefix}_rprod"
    shifted = f"{prefix}_rsh"
    lines = [
        sign_extend_wire(acc_name, acc_bits, ext64, 64),
        f"    wire signed [63:0] {prod} = {ext64} * {slit(64, mult)};",
        f"    wire signed [63:0] {shifted} = {prod} >>> {shift};",
    ]
    return lines, shifted


def saturate_relu(src_name: str, src_bits: int,
                  dst_name: str, dst_bits: int) -> List[str]:
    """Saturate *src* to signed int{dst_bits} range, clamping negatives to 0."""
    qmax = 2 ** (dst_bits - 1) - 1
    return [
        f"    wire signed [{dst_bits - 1}:0] {dst_name} =",
        f"        ({src_name} > {slit(src_bits, qmax)}) ? {slit(dst_bits, qmax)} :",
        f"        ({src_name} < {slit(src_bits, 0)})   ? {dst_bits}'sd0 :",
        f"        {src_name}[{dst_bits - 1}:0];",
    ]


def saturate_linear(src_name: str, src_bits: int,
                    dst_name: str, dst_bits: int) -> List[str]:
    """Saturate *src* to signed int{dst_bits} range (no activation)."""
    qmax = 2 ** (dst_bits - 1) - 1
    return [
        f"    wire signed [{dst_bits - 1}:0] {dst_name} =",
        f"        ({src_name} > {slit(src_bits, qmax)})  ? {slit(dst_bits, qmax)} :",
        f"        ({src_name} < {slit(src_bits, -qmax)}) ? {slit(dst_bits, -qmax)} :",
        f"        {src_name}[{dst_bits - 1}:0];",
    ]


def saturate(src_name: str, src_bits: int,
             dst_name: str, dst_bits: int,
             activation: str = "none") -> List[str]:
    """Dispatch to the right saturation variant."""
    if activation == "relu":
        return saturate_relu(src_name, src_bits, dst_name, dst_bits)
    return saturate_linear(src_name, src_bits, dst_name, dst_bits)


# ---------------------------------------------------------------------------
#  Accumulator sizing
# ---------------------------------------------------------------------------

def acc_bits_for(n_inputs: int, weight_bits: int = 8) -> int:
    """Minimum accumulator width to avoid overflow."""
    product_bits = 2 * weight_bits
    sum_bits = math.ceil(math.log2(max(n_inputs, 2)))
    return product_bits + sum_bits + 2   # +2 safety margin


# ---------------------------------------------------------------------------
#  Piecewise-linear approximation (for GELU, sigmoid, tanh, SiLU)
# ---------------------------------------------------------------------------

def pwl_lut_lines(
    input_wire: str,
    output_wire: str,
    breakpoints: List[int],
    slopes: List[int],
    offsets: List[int],
    input_bits: int = 8,
    output_bits: int = 8,
    lut_prefix: str = "pwl",
) -> List[str]:
    """
    Generate piecewise-linear approximation using a case-style LUT.

    For an activation f(x), we approximate with N linear segments:
        if x < breakpoints[0]:          y = slopes[0]*x + offsets[0]
        elif x < breakpoints[1]:        y = slopes[1]*x + offsets[1]
        ...
        else:                           y = slopes[-1]*x + offsets[-1]

    All values are in quantized integer space.  slopes and offsets are
    pre-scaled so that:  y_int = (slope * x_int + offset) >> FRAC_BITS

    *breakpoints* has len N-1, *slopes* and *offsets* have len N.
    """
    lines = []
    acc_bits = 24  # enough for slope*input + offset
    lines.append(f"    // Piecewise-linear approximation ({len(slopes)} segments)")
    lines.append(f"    wire signed [{acc_bits - 1}:0] {lut_prefix}_val;")

    # Build nested ternary
    parts = []
    for i, bp in enumerate(breakpoints):
        parts.append(
            f"({input_wire} < {slit(input_bits, bp)}) "
            f"? ({slit(acc_bits, slopes[i])} * {sign_extend_expr(input_wire, input_bits, acc_bits)} "
            f"+ {slit(acc_bits, offsets[i])}) :"
        )
    # Last segment (else)
    parts.append(
        f"({slit(acc_bits, slopes[-1])} * {sign_extend_expr(input_wire, input_bits, acc_bits)} "
        f"+ {slit(acc_bits, offsets[-1])})"
    )

    lines.append(f"    assign {lut_prefix}_val =")
    for p in parts:
        lines.append(f"        {p}")
    lines[-1] += ";"

    # Shift result back to output range (slopes are in Q8 fixed-point)
    shifted = f"{lut_prefix}_shifted"
    lines.append(f"    wire signed [{acc_bits - 1}:0] {shifted} = {lut_prefix}_val >>> 8;")

    qmax = 2 ** (output_bits - 1) - 1
    lines.append(f"    wire signed [{output_bits - 1}:0] {output_wire} =")
    lines.append(f"        ({shifted} > {slit(acc_bits, qmax)})  ? {slit(output_bits, qmax)} :")
    lines.append(f"        ({shifted} < {slit(acc_bits, -qmax)}) ? {slit(output_bits, -qmax)} :")
    lines.append(f"        {shifted}[{output_bits - 1}:0];")

    return lines


# ---------------------------------------------------------------------------
#  Module boilerplate
# ---------------------------------------------------------------------------

def module_header(name: str, in_wires: List[Tuple[str, int]],
                  out_wires: List[Tuple[str, int]]) -> List[str]:
    """
    Generate module declaration.

    in_wires / out_wires:  list of (wire_name, bit_width).
    """
    lines = [f"module {name} ("]
    ports = []
    for wn, bw in in_wires:
        ports.append(f"    input  wire signed [{bw - 1}:0] {wn}")
    for wn, bw in out_wires:
        ports.append(f"    output wire signed [{bw - 1}:0] {wn}")
    lines.append(",\n".join(ports))
    lines.append(");")
    return lines


def module_footer() -> List[str]:
    return ["", "endmodule"]


def section_comment(text: str, width: int = 71) -> List[str]:
    return [
        f"    // {'=' * width}",
        f"    // {text}",
        f"    // {'=' * width}",
    ]
