"""
estimate.py -- Pre-synthesis area/resource estimator for weights2silicon.

Answers "Does this fit on a Tiny Tapeout tile?" before you spend time
running Yosys.  Walks the compute graph topologically, counts operations,
and produces LUT/FF/gate estimates for both combinational and sequential
architectures.

Usage:
    from w2s.estimate import estimate
    report = estimate(graph, mode='sequential')
    print(report)
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

from w2s.core import ComputeGraph, Operation, OpType


# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

# Combinational mode: empirical LUT costs (8-bit data path)
_COMB_LUTS_PER_CONST_MUL_8B = 24     # avg shift-add for random 8-bit constant
_COMB_LUTS_PER_ADDER_8B = 8           # 1 LUT per bit for ripple-carry
_COMB_LUTS_PER_REQUANT = 64           # 64-bit multiply + shift
_COMB_LUTS_PER_SATURATION = 8         # clamp + ReLU

# Sequential mode: fixed hardware costs
_SEQ_LUTS_MULTIPLIER = 200            # 8x8 signed multiplier
_SEQ_LUTS_ACCUMULATOR = 32            # 32-bit accumulator
_SEQ_LUTS_REQUANT = 128               # 64-bit requantization multiply
_SEQ_LUTS_CONTROL = 50                # state machine + counters

# Tiny Tapeout tile sizes
_TT_TILE_LUTS = 160                   # ~1000 std cells ~ 160 LUTs
_TT_GATES_PER_LUT = 6.25             # ~1000 gates / 160 LUTs
_TT_MAX_TILES = 16                    # 8x2 = 16 tiles
_TT_MAX_LUTS = _TT_TILE_LUTS * _TT_MAX_TILES  # 2560


# ---------------------------------------------------------------------------
#  Tile configurations
# ---------------------------------------------------------------------------

_TT_CONFIGS = [
    ("1x1", 1, 160),
    ("1x2", 2, 320),
    ("2x2", 4, 640),
    ("4x2", 8, 1280),
    ("8x2", 16, 2560),
]


# ---------------------------------------------------------------------------
#  EstimateReport
# ---------------------------------------------------------------------------

@dataclass
class EstimateReport:
    """Area, resource, and timing estimates for a w2s design."""
    total_params: int
    total_multipliers: int
    total_adders: int
    total_mux_bits: int
    estimated_luts: int
    estimated_ffs: int
    rom_bits: int
    buffer_bits: int
    mode: str
    cycles_per_inference: int
    fits_tiny_tapeout: bool
    tt_tiles_needed: int
    warnings: List[str] = field(default_factory=list)

    # Optional breakdown for pretty-printing
    _breakdown: Dict[str, int] = field(default_factory=dict, repr=False)

    def __str__(self) -> str:
        gate_equiv = int(self.estimated_luts * _TT_GATES_PER_LUT)
        lines = []

        lines.append(f"=== Area Estimate ({self.mode} mode) ===")
        lines.append(f"Parameters:      {self.total_params:,} "
                     f"({self.rom_bits:,} bits)")

        if self.mode == "sequential":
            rom_luts = self._breakdown.get("rom_luts", 0)
            mac_luts = self._breakdown.get("mac_luts", 0)
            ctrl_luts = self._breakdown.get("control_luts", 0)
            lines.append(f"Weight ROM:      ~{rom_luts:,} LUTs")
            lines.append(f"MAC engine:      ~{mac_luts:,} LUTs")
            lines.append(f"Buffers:         {self.estimated_ffs:,} FFs")
            lines.append(f"Control:         ~{ctrl_luts:,} LUTs")
        else:
            mul_luts = self._breakdown.get("multiplier_luts", 0)
            add_luts = self._breakdown.get("adder_luts", 0)
            rq_luts = self._breakdown.get("requant_luts", 0)
            sat_luts = self._breakdown.get("saturation_luts", 0)
            lines.append(f"Multipliers:     {self.total_multipliers:,} "
                         f"(~{mul_luts:,} LUTs)")
            lines.append(f"Adders:          {self.total_adders:,} "
                         f"(~{add_luts:,} LUTs)")
            lines.append(f"Requantization:  ~{rq_luts:,} LUTs")
            lines.append(f"Saturation/ReLU: ~{sat_luts:,} LUTs")

        lines.append(f"Total:           ~{self.estimated_luts:,} LUTs "
                     f"(~{gate_equiv:,} gates)")
        lines.append("")

        lines.append(f"Cycles/inference: {self.cycles_per_inference:,}")
        if self.cycles_per_inference > 0:
            freq_mhz = 100
            throughput = freq_mhz * 1e6 / self.cycles_per_inference
            if throughput >= 1e6:
                tp_str = f"~{throughput / 1e6:.1f}M inferences/sec @ {freq_mhz}MHz"
            elif throughput >= 1e3:
                tp_str = f"~{throughput / 1e3:.1f}K inferences/sec @ {freq_mhz}MHz"
            else:
                tp_str = f"~{throughput:.0f} inferences/sec @ {freq_mhz}MHz"
            lines.append(f"Throughput:       {tp_str}")
        lines.append("")

        if self.fits_tiny_tapeout:
            rec = _recommend_tile(self.estimated_luts)
            lines.append(f"Tiny Tapeout:    FITS "
                         f"({self.tt_tiles_needed} tile{'s' if self.tt_tiles_needed != 1 else ''}"
                         f" of {_TT_TILE_LUTS} LUTs each)")
            lines.append(f"                 Recommended: TT {rec} tile")
        else:
            lines.append(f"Tiny Tapeout:    DOES NOT FIT")
            lines.append(f"                 Needs ~{self.tt_tiles_needed} tiles "
                         f"(max {_TT_MAX_TILES} tiles = "
                         f"{_TT_MAX_LUTS:,} LUTs)")

        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")

        return "\n".join(lines)


def _recommend_tile(luts: int) -> str:
    """Return the smallest TT tile config that fits."""
    for name, _tiles, capacity in _TT_CONFIGS:
        if luts <= capacity:
            return name
    return "8x2 (insufficient)"


# ---------------------------------------------------------------------------
#  Main estimator
# ---------------------------------------------------------------------------

def estimate(graph: ComputeGraph, mode: str = "combinational") -> EstimateReport:
    """
    Estimate area, resources, and timing for a design.

    Args:
        graph: A ComputeGraph (quantized or unquantized -- uses q_weights
               if available, falls back to float weights for param counts).
        mode:  'combinational' or 'sequential'.

    Returns:
        An EstimateReport with LUT/FF/gate estimates and Tiny Tapeout fit.
    """
    if mode not in ("combinational", "sequential"):
        raise ValueError(f"Unknown mode '{mode}'; expected 'combinational' or 'sequential'")

    bits = graph.quant_config.bits
    ops = graph.topological_order()
    warnings: List[str] = []

    # --- Count parameters and collect layer info ---
    total_params = 0
    layer_info: List[Dict] = []  # per-op metadata for estimation

    for op in ops:
        info = _analyze_op(op, bits)
        layer_info.append(info)
        total_params += info["n_params"]

    rom_bits = total_params * bits

    if mode == "combinational":
        report = _estimate_combinational(ops, layer_info, bits, total_params,
                                         rom_bits, warnings)
    else:
        report = _estimate_sequential(ops, layer_info, bits, total_params,
                                      rom_bits, warnings)

    return report


# ---------------------------------------------------------------------------
#  Per-op analysis
# ---------------------------------------------------------------------------

def _analyze_op(op: Operation, bits: int) -> Dict:
    """Extract key metrics from a single operation for estimation."""
    # Prefer quantized weights, fall back to float
    weights = op.q_weights if op.q_weights else op.weights

    n_params = sum(int(np.prod(w.shape)) for w in weights.values())

    info = {
        "op": op,
        "op_type": op.op_type,
        "name": op.name,
        "n_params": n_params,
        "n_in": 0,
        "n_out": 0,
        "n_multiplies": 0,
        "n_additions": 0,
        "n_neurons": 0,
        "output_size": 0,
    }

    if op.op_type == OpType.DENSE:
        w = weights.get("weight")
        if w is not None:
            n_out, n_in = w.shape
            info["n_in"] = n_in
            info["n_out"] = n_out
            info["n_multiplies"] = n_in * n_out
            info["n_additions"] = (n_in - 1) * n_out  # adder tree per neuron
            info["n_neurons"] = n_out
            info["output_size"] = n_out

    elif op.op_type == OpType.CONV2D:
        w = weights.get("weight")
        if w is not None:
            c_out, c_in, kh, kw = w.shape
            kernel_elems = c_in * kh * kw
            # Output spatial size depends on input -- estimate from attrs
            stride = op.attrs.get("stride", (1, 1))
            padding = op.attrs.get("padding", (0, 0))
            # Without knowing input spatial dims exactly, estimate based on
            # a common scenario.  The param-based cost is dominant anyway.
            info["n_in"] = kernel_elems
            info["n_out"] = c_out
            info["n_multiplies"] = n_params - c_out  # exclude bias
            info["n_additions"] = (kernel_elems - 1) * c_out
            info["n_neurons"] = c_out
            info["output_size"] = c_out

    elif op.op_type == OpType.CONV1D:
        w = weights.get("weight")
        if w is not None:
            c_out, c_in, k = w.shape
            kernel_elems = c_in * k
            info["n_in"] = kernel_elems
            info["n_out"] = c_out
            info["n_multiplies"] = n_params - c_out
            info["n_additions"] = (kernel_elems - 1) * c_out
            info["n_neurons"] = c_out
            info["output_size"] = c_out

    elif op.op_type == OpType.EMBEDDING:
        w = weights.get("weight")
        if w is not None:
            vocab, dim = w.shape
            info["n_in"] = 1
            info["n_out"] = dim
            info["output_size"] = dim

    elif op.op_type == OpType.MULTI_HEAD_ATTENTION:
        # 4 dense projections: Q, K, V, Out
        for key in ("q_weight", "k_weight", "v_weight", "out_weight"):
            w = weights.get(key)
            if w is not None:
                n_out_w, n_in_w = w.shape
                info["n_multiplies"] += n_in_w * n_out_w
                info["n_additions"] += (n_in_w - 1) * n_out_w
        embed_dim = op.attrs.get("embed_dim", 0)
        info["n_in"] = embed_dim
        info["n_out"] = embed_dim
        info["n_neurons"] = embed_dim * 4  # Q + K + V + Out projections
        info["output_size"] = embed_dim

    elif op.op_type in (OpType.LAYERNORM, OpType.RMSNORM, OpType.BATCHNORM):
        info["n_neurons"] = n_params  # scale + bias elements
        info["output_size"] = n_params

    elif op.op_type in (OpType.RELU, OpType.GELU, OpType.SIGMOID,
                        OpType.TANH, OpType.SILU, OpType.SOFTMAX):
        # Element-wise: cost proportional to input size, no weight params
        pass

    return info


# ---------------------------------------------------------------------------
#  Combinational estimate
# ---------------------------------------------------------------------------

def _estimate_combinational(
    ops: List[Operation],
    layer_info: List[Dict],
    bits: int,
    total_params: int,
    rom_bits: int,
    warnings: List[str],
) -> EstimateReport:
    """Estimate for fully-unrolled combinational (no clock) design."""

    total_multipliers = 0
    total_adders = 0

    multiplier_luts = 0
    adder_luts = 0
    requant_luts = 0
    saturation_luts = 0

    # Scale LUT costs linearly with bit width relative to 8-bit baseline
    bit_scale = bits / 8.0

    for info in layer_info:
        op_type = info["op_type"]

        if op_type in (OpType.DENSE, OpType.CONV2D, OpType.CONV1D,
                       OpType.MULTI_HEAD_ATTENTION):
            n_mul = info["n_multiplies"]
            n_add = info["n_additions"]
            n_neurons = info["n_neurons"]

            total_multipliers += n_mul
            total_adders += n_add

            # Each const multiply: ~24 LUTs at 8-bit, scales with bit width
            multiplier_luts += int(n_mul * _COMB_LUTS_PER_CONST_MUL_8B * bit_scale)
            # Each addition: ~8 LUTs at 8-bit
            adder_luts += int(n_add * _COMB_LUTS_PER_ADDER_8B * bit_scale)
            # Requantization per neuron
            requant_luts += int(n_neurons * _COMB_LUTS_PER_REQUANT)
            # Saturation/clamp per neuron
            saturation_luts += int(n_neurons * _COMB_LUTS_PER_SATURATION)

        elif op_type in (OpType.RELU, OpType.SIGMOID, OpType.TANH,
                         OpType.SILU, OpType.GELU):
            # Activation functions: element-wise, cost depends on type
            if op_type == OpType.RELU:
                # ReLU is just a mux: ~2 LUTs per element
                pass  # accounted for in saturation_luts above for dense/conv
            else:
                # PWL approximation: ~32 LUTs per element (comparators + muxes)
                # We don't know the exact tensor size without shape inference,
                # so account for it as a warning
                warnings.append(
                    f"Standalone {op_type.value} activation cost not estimated "
                    f"(folded into preceding layer when possible)")

        elif op_type == OpType.EMBEDDING:
            # Embedding is a ROM lookup: ~1 LUT per bit stored
            n_p = info["n_params"]
            multiplier_luts += int(n_p * bits)  # ROM as LUT fabric

        elif op_type in (OpType.LAYERNORM, OpType.RMSNORM, OpType.BATCHNORM):
            # Normalization requires division (expensive in hardware)
            n_neurons = info["n_neurons"]
            requant_luts += int(n_neurons * _COMB_LUTS_PER_REQUANT * 2)
            warnings.append(
                f"{info['name']}: normalization layers are expensive in "
                f"combinational mode (~{n_neurons * _COMB_LUTS_PER_REQUANT * 2} LUTs)")

    estimated_luts = multiplier_luts + adder_luts + requant_luts + saturation_luts

    # Combinational: no flip-flops, no ROM (weights are constants)
    estimated_ffs = 0
    total_mux_bits = 0  # no muxing needed in combinational
    cycles = 1

    tt_tiles = math.ceil(estimated_luts / _TT_TILE_LUTS) if estimated_luts > 0 else 1
    fits = estimated_luts <= _TT_MAX_LUTS

    if estimated_luts > _TT_MAX_LUTS:
        warnings.append(
            f"Design exceeds TT 8x2 max ({estimated_luts:,} LUTs > "
            f"{_TT_MAX_LUTS:,} LUTs). Consider sequential mode.")

    if total_params > 500 and bits >= 8:
        warnings.append(
            "Large combinational design: synthesis may be slow. "
            "Consider sequential mode for smaller area.")

    return EstimateReport(
        total_params=total_params,
        total_multipliers=total_multipliers,
        total_adders=total_adders,
        total_mux_bits=total_mux_bits,
        estimated_luts=estimated_luts,
        estimated_ffs=estimated_ffs,
        rom_bits=rom_bits,
        buffer_bits=0,
        mode="combinational",
        cycles_per_inference=cycles,
        fits_tiny_tapeout=fits,
        tt_tiles_needed=tt_tiles,
        warnings=warnings,
        _breakdown={
            "multiplier_luts": multiplier_luts,
            "adder_luts": adder_luts,
            "requant_luts": requant_luts,
            "saturation_luts": saturation_luts,
        },
    )


# ---------------------------------------------------------------------------
#  Sequential estimate
# ---------------------------------------------------------------------------

def _estimate_sequential(
    ops: List[Operation],
    layer_info: List[Dict],
    bits: int,
    total_params: int,
    rom_bits: int,
    warnings: List[str],
) -> EstimateReport:
    """Estimate for time-multiplexed sequential (clocked) design."""

    # Fixed MAC hardware
    mac_luts = _SEQ_LUTS_MULTIPLIER + _SEQ_LUTS_ACCUMULATOR + _SEQ_LUTS_REQUANT
    control_luts = _SEQ_LUTS_CONTROL

    # Weight ROM: case statement in LUT fabric
    # ~1 LUT per bit stored (Yosys maps case to LUT tree)
    rom_luts = rom_bits  # 1 LUT per bit

    # Activation buffers: need to hold the largest intermediate tensor
    # between layers so we can feed it to the next layer
    max_buffer_size = 0
    total_cycles = 0
    total_mux_bits = 0

    for info in layer_info:
        op_type = info["op_type"]
        out_size = info["output_size"]

        if out_size > max_buffer_size:
            max_buffer_size = out_size

        if op_type in (OpType.DENSE, OpType.CONV2D, OpType.CONV1D,
                       OpType.MULTI_HEAD_ATTENTION):
            n_in = info["n_in"]
            n_out = info["n_out"]
            if n_in > 0 and n_out > 0:
                total_cycles += n_in * n_out
            # Mux bits for selecting weights/inputs
            total_mux_bits += max(
                math.ceil(math.log2(max(info["n_params"], 2))),
                0
            ) * bits

        elif op_type == OpType.EMBEDDING:
            # Embedding lookup: 1 cycle per output element
            total_cycles += info["n_out"]

    # Also check graph input sizes for buffer
    for inp_size in graph_input_sizes(ops):
        if inp_size > max_buffer_size:
            max_buffer_size = inp_size

    buffer_bits = max_buffer_size * bits
    buffer_ffs = buffer_bits  # 1 FF per bit

    # State register FFs: address counters, FSM state, accumulator
    state_ffs = 32 + 16 + 32  # addr + state + acc register
    estimated_ffs = buffer_ffs + state_ffs

    estimated_luts = rom_luts + mac_luts + control_luts

    # If cycles is 0 (no dense/conv layers), set minimum
    if total_cycles == 0:
        total_cycles = 1

    tt_tiles = math.ceil(estimated_luts / _TT_TILE_LUTS) if estimated_luts > 0 else 1
    fits = estimated_luts <= _TT_MAX_LUTS

    if rom_luts > _TT_MAX_LUTS:
        warnings.append(
            f"Weight ROM alone needs ~{rom_luts:,} LUTs "
            f"(exceeds TT 8x2 max of {_TT_MAX_LUTS:,}). "
            f"Consider external SPI flash or smaller model.")

    if estimated_luts > _TT_MAX_LUTS:
        warnings.append(
            f"Design exceeds TT 8x2 max ({estimated_luts:,} LUTs > "
            f"{_TT_MAX_LUTS:,} LUTs). Reduce parameters or bit width.")

    return EstimateReport(
        total_params=total_params,
        total_multipliers=1,  # sequential: one shared multiplier
        total_adders=1,       # one shared accumulator
        total_mux_bits=total_mux_bits,
        estimated_luts=estimated_luts,
        estimated_ffs=estimated_ffs,
        rom_bits=rom_bits,
        buffer_bits=buffer_bits,
        mode="sequential",
        cycles_per_inference=total_cycles,
        fits_tiny_tapeout=fits,
        tt_tiles_needed=tt_tiles,
        warnings=warnings,
        _breakdown={
            "rom_luts": rom_luts,
            "mac_luts": mac_luts,
            "control_luts": control_luts,
        },
    )


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def graph_input_sizes(ops: List[Operation]) -> List[int]:
    """Estimate input tensor sizes from the first layer's n_in."""
    sizes = []
    for info_op in ops:
        if info_op.op_type in (OpType.DENSE, OpType.CONV2D, OpType.CONV1D):
            weights = info_op.q_weights if info_op.q_weights else info_op.weights
            w = weights.get("weight")
            if w is not None:
                if info_op.op_type == OpType.DENSE:
                    sizes.append(w.shape[1])  # n_in
                elif info_op.op_type in (OpType.CONV2D, OpType.CONV1D):
                    sizes.append(int(np.prod(w.shape[1:])))
            break  # only need first layer
    return sizes
