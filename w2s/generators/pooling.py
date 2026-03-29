"""
pooling.py -- Verilog generators for MaxPool2D, AvgPool2D, and GlobalAvgPool.

MaxPool uses pairwise comparator trees to find the maximum value in each
kernel window.  AvgPool sums the window and divides (right-shift for
power-of-two kernel sizes, reciprocal multiply otherwise).  GlobalAvgPool
averages every spatial position per channel.
"""

import math
from typing import Dict, List, Tuple

from w2s.core import Operation, TensorWires
from w2s.emit import (
    section_comment,
    wire_signed,
    sign_extend_wire,
    slit,
    saturate_linear,
    acc_bits_for,
)


# ---------------------------------------------------------------------------
#  Comparator tree helper
# ---------------------------------------------------------------------------

def _max_tree(
    wires: List[str],
    bits: int,
    prefix: str,
    lines: List[str],
) -> str:
    """
    Build a balanced binary comparator tree over *wires* and return the
    name of the final max wire.  Intermediate wires are appended to *lines*.

    Each comparison:  wire signed [b-1:0] max_X = (a > b) ? a : b;
    """
    if len(wires) == 0:
        # Degenerate: should not happen, but return zero for safety
        z = f"{prefix}_zero"
        lines.append(wire_signed(z, bits))
        lines.append(f"    assign {z} = {slit(bits, 0)};")
        return z

    if len(wires) == 1:
        return wires[0]

    level = 0
    current = list(wires)
    while len(current) > 1:
        nxt: List[str] = []
        for i in range(0, len(current), 2):
            if i + 1 < len(current):
                a, b = current[i], current[i + 1]
                mx = f"{prefix}_l{level}_m{i // 2}"
                lines.append(
                    f"    wire signed [{bits - 1}:0] {mx} = "
                    f"({a} > {b}) ? {a} : {b};"
                )
                nxt.append(mx)
            else:
                # Odd element passes through
                nxt.append(current[i])
        current = nxt
        level += 1

    return current[0]


# ---------------------------------------------------------------------------
#  MaxPool2D
# ---------------------------------------------------------------------------

def generate_maxpool2d(
    op: Operation,
    wire_map: Dict[str, TensorWires],
    bits: int,
) -> Tuple[List[str], Dict[str, TensorWires]]:
    """
    Generate Verilog for 2-D max pooling.

    For each output position, a comparator tree selects the maximum
    value from the kernel window.

    Input shape  : (C, H_in, W_in)
    Output shape : (C, H_out, W_out)

    Returns (verilog_lines, new_wires).
    """
    # ---- unpack --------------------------------------------------------
    kH, kW = op.attrs['kernel_size']
    sH, sW = op.attrs['stride']

    inp_name = op.inputs[0]
    inp_tw = wire_map[inp_name]
    C, H_in, W_in = inp_tw.shape

    H_out = (H_in - kH) // sH + 1
    W_out = (W_in - kW) // sW + 1

    out_name = op.outputs[0]
    n_out = C * H_out * W_out
    out_wire_names = [f"{op.name}_out_{i}" for i in range(n_out)]

    lines: List[str] = []

    lines += section_comment(
        f"{op.name}: MaxPool2D  in({C},{H_in},{W_in}) "
        f"k({kH},{kW}) s({sH},{sW}) -> out({C},{H_out},{W_out})"
    )
    lines.append("")

    # ---- per-output-position -------------------------------------------
    for c in range(C):
        for oh in range(H_out):
            for ow in range(W_out):
                flat_idx = c * H_out * W_out + oh * W_out + ow
                prefix = f"{op.name}_p{flat_idx}"

                lines.append(
                    f"    // --- output[{c},{oh},{ow}] (flat {flat_idx}) ---"
                )

                # Collect input wires in the kernel window
                window_wires: List[str] = []
                for r in range(kH):
                    for s in range(kW):
                        ih = oh * sH + r
                        iw = ow * sW + s
                        inp_flat = c * H_in * W_in + ih * W_in + iw
                        window_wires.append(inp_tw.wire_names[inp_flat])

                # Build comparator tree
                result = _max_tree(window_wires, bits, prefix, lines)

                # Assign output wire
                out_wire = out_wire_names[flat_idx]
                if result == out_wire:
                    pass  # already named correctly (single element)
                else:
                    lines.append(
                        f"    wire signed [{bits - 1}:0] {out_wire} = {result};"
                    )
                lines.append("")

    # ---- build output TensorWires --------------------------------------
    out_tw = TensorWires(
        wire_names=out_wire_names,
        shape=(C, H_out, W_out),
        bits=bits,
        signed=True,
    )

    return lines, {out_name: out_tw}


# ---------------------------------------------------------------------------
#  AvgPool2D
# ---------------------------------------------------------------------------

def generate_avgpool2d(
    op: Operation,
    wire_map: Dict[str, TensorWires],
    bits: int,
) -> Tuple[List[str], Dict[str, TensorWires]]:
    """
    Generate Verilog for 2-D average pooling.

    Sums every kernel window and divides by the window size.  If the
    window size is a power of two the division is a right-shift;
    otherwise it is an integer multiply by the reciprocal constant.

    Input shape  : (C, H_in, W_in)
    Output shape : (C, H_out, W_out)

    Returns (verilog_lines, new_wires).
    """
    # ---- unpack --------------------------------------------------------
    kH, kW = op.attrs['kernel_size']
    sH, sW = op.attrs['stride']

    inp_name = op.inputs[0]
    inp_tw = wire_map[inp_name]
    C, H_in, W_in = inp_tw.shape

    H_out = (H_in - kH) // sH + 1
    W_out = (W_in - kW) // sW + 1

    window_size = kH * kW
    abits = acc_bits_for(window_size, bits)

    out_name = op.outputs[0]
    n_out = C * H_out * W_out
    out_wire_names = [f"{op.name}_out_{i}" for i in range(n_out)]

    lines: List[str] = []

    # Determine division strategy
    is_pow2 = (window_size & (window_size - 1)) == 0 and window_size > 0
    if is_pow2:
        shift_amt = int(math.log2(window_size))
        div_comment = f"div by {window_size} = >>>{shift_amt}"
    else:
        # Reciprocal in Q16 fixed-point: round(2^16 / window_size)
        recip_shift = 16
        recip_mult = round((1 << recip_shift) / window_size)
        div_comment = (
            f"div by {window_size} via reciprocal "
            f"(mult={recip_mult}, shift={recip_shift})"
        )

    lines += section_comment(
        f"{op.name}: AvgPool2D  in({C},{H_in},{W_in}) "
        f"k({kH},{kW}) s({sH},{sW}) -> out({C},{H_out},{W_out})"
    )
    lines.append(f"    // {div_comment}")
    lines.append("")

    # ---- per-output-position -------------------------------------------
    for c in range(C):
        for oh in range(H_out):
            for ow in range(W_out):
                flat_idx = c * H_out * W_out + oh * W_out + ow
                prefix = f"{op.name}_p{flat_idx}"

                lines.append(
                    f"    // --- output[{c},{oh},{ow}] (flat {flat_idx}) ---"
                )

                # 1) Sign-extend inputs to accumulator width
                ext_names: List[str] = []
                for r in range(kH):
                    for s in range(kW):
                        ih = oh * sH + r
                        iw = ow * sW + s
                        inp_flat = c * H_in * W_in + ih * W_in + iw
                        src_wire = inp_tw.wire_names[inp_flat]
                        ext = f"{prefix}_ext_{r}_{s}"
                        lines.append(
                            sign_extend_wire(src_wire, bits, ext, abits)
                        )
                        ext_names.append(ext)

                # 2) Sum all values
                sum_name = f"{prefix}_sum"
                lines.append(f"    wire signed [{abits - 1}:0] {sum_name} =")
                for ti, en in enumerate(ext_names):
                    sep = "  " if ti == 0 else "+ "
                    end = ";" if ti == len(ext_names) - 1 else ""
                    lines.append(f"        {sep}{en}{end}")

                # 3) Divide
                div_name = f"{prefix}_div"
                if is_pow2:
                    lines.append(
                        f"    wire signed [{abits - 1}:0] {div_name} = "
                        f"{sum_name} >>> {shift_amt};"
                    )
                else:
                    prod_name = f"{prefix}_rprod"
                    lines.append(
                        f"    wire signed [{abits + 16 - 1}:0] {prod_name} = "
                        f"{sum_name} * {slit(abits, recip_mult)};"
                    )
                    lines.append(
                        f"    wire signed [{abits - 1}:0] {div_name} = "
                        f"{prod_name} >>> {recip_shift};"
                    )

                # 4) Saturate back to target bit width
                out_wire = out_wire_names[flat_idx]
                lines += saturate_linear(div_name, abits, out_wire, bits)
                lines.append("")

    # ---- build output TensorWires --------------------------------------
    out_tw = TensorWires(
        wire_names=out_wire_names,
        shape=(C, H_out, W_out),
        bits=bits,
        signed=True,
    )

    return lines, {out_name: out_tw}


# ---------------------------------------------------------------------------
#  Global Average Pool
# ---------------------------------------------------------------------------

def generate_global_avgpool(
    op: Operation,
    wire_map: Dict[str, TensorWires],
    bits: int,
) -> Tuple[List[str], Dict[str, TensorWires]]:
    """
    Generate Verilog for global average pooling.

    Averages all spatial positions per channel.
    Input shape  : (C, H, W)
    Output shape : (C,)

    Returns (verilog_lines, new_wires).
    """
    # ---- unpack --------------------------------------------------------
    inp_name = op.inputs[0]
    inp_tw = wire_map[inp_name]
    C = inp_tw.shape[0]
    H = inp_tw.shape[1]
    W = inp_tw.shape[2]
    spatial = H * W

    abits = acc_bits_for(spatial, bits)

    out_name = op.outputs[0]
    out_wire_names = [f"{op.name}_out_{c}" for c in range(C)]

    lines: List[str] = []

    # Division strategy
    is_pow2 = (spatial & (spatial - 1)) == 0 and spatial > 0
    if is_pow2:
        shift_amt = int(math.log2(spatial))
        div_comment = f"div by {spatial} = >>>{shift_amt}"
    else:
        recip_shift = 16
        recip_mult = round((1 << recip_shift) / spatial)
        div_comment = (
            f"div by {spatial} via reciprocal "
            f"(mult={recip_mult}, shift={recip_shift})"
        )

    lines += section_comment(
        f"{op.name}: GlobalAvgPool  in({C},{H},{W}) -> out({C},)"
    )
    lines.append(f"    // {div_comment}")
    lines.append("")

    # ---- per-channel ---------------------------------------------------
    for c in range(C):
        prefix = f"{op.name}_c{c}"

        lines.append(f"    // --- channel {c} ---")

        # 1) Sign-extend all spatial inputs
        ext_names: List[str] = []
        for h in range(H):
            for w in range(W):
                inp_flat = c * H * W + h * W + w
                src_wire = inp_tw.wire_names[inp_flat]
                ext = f"{prefix}_ext_{h}_{w}"
                lines.append(
                    sign_extend_wire(src_wire, bits, ext, abits)
                )
                ext_names.append(ext)

        # 2) Sum
        sum_name = f"{prefix}_sum"
        lines.append(f"    wire signed [{abits - 1}:0] {sum_name} =")
        for ti, en in enumerate(ext_names):
            sep = "  " if ti == 0 else "+ "
            end = ";" if ti == len(ext_names) - 1 else ""
            lines.append(f"        {sep}{en}{end}")

        # 3) Divide
        div_name = f"{prefix}_div"
        if is_pow2:
            lines.append(
                f"    wire signed [{abits - 1}:0] {div_name} = "
                f"{sum_name} >>> {shift_amt};"
            )
        else:
            prod_name = f"{prefix}_rprod"
            lines.append(
                f"    wire signed [{abits + 16 - 1}:0] {prod_name} = "
                f"{sum_name} * {slit(abits, recip_mult)};"
            )
            lines.append(
                f"    wire signed [{abits - 1}:0] {div_name} = "
                f"{prod_name} >>> {recip_shift};"
            )

        # 4) Saturate
        out_wire = out_wire_names[c]
        lines += saturate_linear(div_name, abits, out_wire, bits)
        lines.append("")

    # ---- build output TensorWires --------------------------------------
    out_tw = TensorWires(
        wire_names=out_wire_names,
        shape=(C,),
        bits=bits,
        signed=True,
    )

    return lines, {out_name: out_tw}
