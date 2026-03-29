"""
embedding.py -- Verilog generator for embedding (lookup table) layers.

Produces a combinational ROM for each embedding dimension.  Given an
integer index, the ROM outputs the corresponding hardwired vector from
the quantized embedding table via a Verilog case-style mux chain.
"""

import numpy as np
from typing import Dict, List, Tuple
import math

from w2s.core import Operation, TensorWires
from w2s.emit import (
    section_comment,
    wire_signed,
    slit,
)


def generate_embedding(
    op: Operation,
    wire_map: Dict[str, TensorWires],
    bits: int,
) -> Tuple[List[str], Dict[str, TensorWires]]:
    """
    Generate Verilog for an embedding lookup table.

    For each input index position and each embedding dimension, a
    combinational ROM is emitted as a chain of ternary comparisons:

        wire signed [7:0] emb_out_0 =
            (idx == 0) ? 8'sd42 :
            (idx == 1) ? -8'sd17 :
            ...
            8'sd0;  // default

    Weight shape : (num_embeddings, embedding_dim) -- quantized integers
    Input        : one or more unsigned index wires (seq_len positions)
    Output shape : (seq_len, embedding_dim)

    Returns (verilog_lines, new_wires).
    """
    # ---- unpack --------------------------------------------------------
    weight = op.q_weights['weight']                     # (V, D) quantized
    V, D = weight.shape

    num_embeddings = op.attrs.get('num_embeddings', V)
    embedding_dim = op.attrs.get('embedding_dim', D)

    # ---- input wires ---------------------------------------------------
    inp_name = op.inputs[0]
    inp_tw = wire_map[inp_name]
    seq_len = inp_tw.numel                              # number of index positions

    # Index bit-width: enough unsigned bits to address V entries
    idx_bits = max(1, math.ceil(math.log2(max(num_embeddings, 2))))

    out_name = op.outputs[0]

    lines: List[str] = []

    # header comment
    lines += section_comment(
        f"{op.name}: Embedding  vocab={num_embeddings}  dim={embedding_dim}  "
        f"seq_len={seq_len}"
    )
    lines.append(f"    // {num_embeddings * embedding_dim} values hardwired as ROM")
    lines.append("")

    # ---- generate ROMs per input position ------------------------------
    out_wire_names: List[str] = []

    for pos in range(seq_len):
        idx_wire = inp_tw.wire_names[pos]

        lines.append(f"    // --- position {pos} ---")

        for d in range(embedding_dim):
            out_wire = f"{op.name}_out_{pos * embedding_dim + d}"
            out_wire_names.append(out_wire)

            # Build ternary mux chain for this dimension
            lines.append(f"    wire signed [{bits - 1}:0] {out_wire} =")
            for v in range(num_embeddings):
                val = int(weight[v, d])
                if v < num_embeddings - 1:
                    lines.append(
                        f"        ({idx_wire} == {v}) ? {slit(bits, val)} :"
                    )
                else:
                    # Last entry doubles as default
                    lines.append(
                        f"        ({idx_wire} == {v}) ? {slit(bits, val)} :"
                    )
                    lines.append(
                        f"        {slit(bits, 0)};  // default"
                    )

        lines.append("")

    # ---- build output TensorWires --------------------------------------
    out_tw = TensorWires(
        wire_names=out_wire_names,
        shape=(seq_len, embedding_dim),
        bits=bits,
        signed=True,
    )

    return lines, {out_name: out_tw}
