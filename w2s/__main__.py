"""
__main__.py -- CLI entry point for weights2silicon.

Usage:
    python -m w2s compile model.onnx --mode sequential --bits 8 --output output/
    python -m w2s estimate model.onnx --mode both
    python -m w2s info model.onnx
"""

import argparse
import sys
import os
import numpy as np
from pathlib import Path

__version__ = "0.2.0"

BANNER = "weights2silicon (w2s) — Compile neural networks to hardwired Verilog"

SUPPORTED_FORMATS = [".onnx", ".safetensors"]


# ---------------------------------------------------------------------------
#  Model loading
# ---------------------------------------------------------------------------

def _load_model(model_path: str, name: str = None):
    """Load a model file and return a ComputeGraph.

    Supports .onnx files.  .safetensors is recognized but not yet fully
    supported.
    """
    path = Path(model_path)

    if not path.exists():
        print(f"Error: file not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    ext = path.suffix.lower()

    if ext == ".onnx":
        try:
            from w2s.importers.onnx_import import load_onnx
        except ImportError:
            print(
                "Error: the 'onnx' package is required to load .onnx models.\n"
                "Install it with:  pip install onnx",
                file=sys.stderr,
            )
            sys.exit(1)
        return load_onnx(str(path), name=name)

    elif ext == ".safetensors":
        print(
            "Error: .safetensors loading requires architecture detection which is\n"
            "not yet fully supported.  For now, please convert your model to ONNX\n"
            "format first.  You can use torch.onnx.export() or optimum-cli:\n"
            "\n"
            "    optimum-cli export onnx --model <hf_model_id> output_dir/\n",
            file=sys.stderr,
        )
        sys.exit(1)

    else:
        print(
            f"Error: unsupported model format '{ext}'.\n"
            f"Supported formats: {', '.join(SUPPORTED_FORMATS)}",
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
#  Parameter counting helpers
# ---------------------------------------------------------------------------

def _count_params(graph):
    """Count total float parameters across all operations."""
    total = 0
    for op in graph.operations:
        for w in op.weights.values():
            total += int(np.prod(w.shape))
    return total


def _count_q_params(graph):
    """Count total quantized parameters across all operations."""
    total = 0
    for op in graph.operations:
        for w in op.q_weights.values():
            total += int(np.prod(w.shape))
    return total


# ---------------------------------------------------------------------------
#  Command: compile
# ---------------------------------------------------------------------------

def cmd_compile(args):
    from w2s.core import QuantConfig, QuantScheme, QuantGranularity

    model_name = args.name or Path(args.model).stem
    print(BANNER)
    print()
    print(f"  Model : {args.model}")
    print(f"  Name  : {model_name}")
    print(f"  Bits  : {args.bits}")
    print(f"  Output: {args.output}")
    print()

    # 1. Load
    print("[1/4] Loading model...")
    graph = _load_model(args.model, name=model_name)
    total_params = _count_params(graph)
    print(f"       {len(graph.operations)} operations, {total_params:,} parameters")

    # 2. Configure quantization
    print(f"[2/4] Quantizing to int{args.bits}...")
    config = QuantConfig(
        bits=args.bits,
        scheme=QuantScheme.SYMMETRIC,
        granularity=QuantGranularity.PER_TENSOR,
    )
    graph.quant_config = config

    # Generate random calibration data and warn the user
    calib_data = {}
    for inp_name in graph.input_names:
        shape = graph.input_shapes.get(inp_name, (1,))
        # Prepend a batch dimension of 4 samples for calibration
        calib_shape = (4,) + tuple(shape)
        calib_data[inp_name] = np.random.randn(*calib_shape).astype(np.float32)

    print("       WARNING: using random calibration data.  For better accuracy,")
    print("       supply real calibration data from your dataset.")

    from w2s.quantize import quantize_graph
    graph = quantize_graph(graph, calib_data, config)

    q_params = _count_q_params(graph)
    print(f"       Quantized {q_params:,} parameters to int{args.bits}")

    # 3. Select mode
    mode = args.mode
    if mode == "auto":
        mode = "sequential" if total_params > 50000 else "combinational"
        print(f"[3/4] Auto-selected mode: {mode} ({total_params:,} params)")
    else:
        print(f"[3/4] Mode: {mode}")

    # 4. Compile
    print("[4/4] Compiling to Verilog...")
    from w2s.graph import compile_graph
    output_path = compile_graph(graph, output_dir=args.output, mode=mode)

    print()
    print("  Done!")
    print(f"  Verilog : {output_path}")
    print(f"  Mode    : {mode}")
    print(f"  Params  : {q_params:,} (all hardwired as constants)")
    print()


# ---------------------------------------------------------------------------
#  Command: estimate
# ---------------------------------------------------------------------------

def cmd_estimate(args):
    print(BANNER)
    print()
    print(f"  Model: {args.model}")
    print(f"  Bits : {args.bits}")
    print()

    print("Loading model...")
    graph = _load_model(args.model)
    total_params = _count_params(graph)

    modes = [args.mode] if args.mode != "both" else ["combinational", "sequential"]

    # Try the estimate module (may not exist yet)
    try:
        from w2s.estimate import estimate as run_estimate
        has_estimator = True
    except ImportError:
        has_estimator = False

    if has_estimator:
        for mode in modes:
            print(f"\n--- {mode} ---")
            report = run_estimate(graph, mode=mode, bits=args.bits)
            print(report)
    else:
        # Fallback: print basic parameter counts
        print("  Note: w2s.estimate module not yet available.")
        print("  Showing basic parameter summary instead.")
        print()
        print(f"  Operations     : {len(graph.operations)}")
        print(f"  Total params   : {total_params:,}")
        print(f"  Quantized bits : {args.bits}")
        print(f"  Weight memory  : {total_params * args.bits / 8 / 1024:.1f} KB (quantized)")
        print()

        for mode in modes:
            print(f"  --- {mode} ---")
            if mode == "combinational":
                # Rough gate estimate: each multiply-accumulate ~ bits^2 gates
                est_gates = total_params * (args.bits ** 2)
                print(f"  Est. logic gates : ~{est_gates:,}")
                print(f"  Est. area (ASIC) : ~{est_gates * 5 / 1e6:.2f} mm^2 (@ 5 um^2/gate, 28nm)")
            else:
                est_gates = 50000 * (args.bits ** 2)
                print(f"  Est. logic gates : ~{est_gates:,} (MAC unit + control)")
                print(f"  Est. SRAM        : ~{total_params * args.bits / 8 / 1024:.1f} KB (weight storage)")
                print(f"  Est. cycles      : ~{total_params:,} (one MAC/cycle)")
            print()


# ---------------------------------------------------------------------------
#  Command: info
# ---------------------------------------------------------------------------

def cmd_info(args):
    print(BANNER)
    print()

    print("Loading model...")
    graph = _load_model(args.model)
    total_params = _count_params(graph)

    print()
    print(f"  Model      : {graph.name}")
    print(f"  Operations : {len(graph.operations)}")
    print(f"  Inputs     : {', '.join(graph.input_names)}")
    print(f"  Outputs    : {', '.join(graph.output_names)}")
    print(f"  Parameters : {total_params:,}")
    print(f"  Est. memory: {total_params * 4 / 1024:.1f} KB (float32)")
    print()

    # Input shapes
    if graph.input_shapes:
        print("  Input shapes:")
        for name, shape in graph.input_shapes.items():
            print(f"    {name}: {shape}")
        print()

    # Layer table
    print("  Layers:")
    print(f"  {'Name':<30} {'Type':<16} {'Params':>10}  Details")
    print(f"  {'-'*30} {'-'*16} {'-'*10}  {'-'*30}")
    for op in graph.operations:
        n_params = sum(int(np.prod(w.shape)) for w in op.weights.values())
        param_str = f"{n_params:,}" if n_params > 0 else "-"

        details = []
        for wname, warr in op.weights.items():
            details.append(f"{wname}{list(warr.shape)}")
        if op.attrs:
            for k, v in op.attrs.items():
                details.append(f"{k}={v}")
        detail_str = ", ".join(details) if details else ""

        print(f"  {op.name:<30} {op.op_type.value:<16} {param_str:>10}  {detail_str}")

    print()
    print(f"  Total: {total_params:,} parameters")
    print()


# ---------------------------------------------------------------------------
#  Argument parser
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        prog="w2s",
        description=BANNER,
    )
    parser.add_argument(
        "--version", action="version", version=f"w2s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- compile ---
    p_compile = subparsers.add_parser(
        "compile",
        help="Compile a neural network model to synthesizable Verilog",
    )
    p_compile.add_argument(
        "model",
        help="Path to model file (.onnx or .safetensors)",
    )
    p_compile.add_argument(
        "--output", "-o",
        default="./output",
        help="Output directory (default: ./output)",
    )
    p_compile.add_argument(
        "--mode", "-m",
        choices=["combinational", "sequential", "auto"],
        default="auto",
        help="Compilation mode (default: auto-select based on model size)",
    )
    p_compile.add_argument(
        "--bits", "-b",
        type=int,
        choices=[4, 8, 16],
        default=8,
        help="Quantization bit width (default: 8)",
    )
    p_compile.add_argument(
        "--name", "-n",
        default=None,
        help="Verilog module name (default: derived from filename)",
    )

    # --- estimate ---
    p_estimate = subparsers.add_parser(
        "estimate",
        help="Estimate area and resource usage without full compilation",
    )
    p_estimate.add_argument(
        "model",
        help="Path to model file (.onnx or .safetensors)",
    )
    p_estimate.add_argument(
        "--mode", "-m",
        choices=["combinational", "sequential", "both"],
        default="both",
        help="Estimation mode (default: both)",
    )
    p_estimate.add_argument(
        "--bits", "-b",
        type=int,
        choices=[4, 8, 16],
        default=8,
        help="Quantization bit width (default: 8)",
    )

    # --- info ---
    p_info = subparsers.add_parser(
        "info",
        help="Show model information (layers, parameters, shapes)",
    )
    p_info.add_argument(
        "model",
        help="Path to model file (.onnx or .safetensors)",
    )

    return parser


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    try:
        if args.command == "compile":
            cmd_compile(args)
        elif args.command == "estimate":
            cmd_estimate(args)
        elif args.command == "info":
            cmd_info(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
