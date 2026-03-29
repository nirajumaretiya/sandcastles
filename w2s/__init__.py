"""
weights2silicon (w2s) — Compile neural network weights to hardwired Verilog.

Public API
----------
    from w2s import ComputeGraph, Operation, OpType, QuantConfig
    from w2s.quantize import quantize
    from w2s.graph import compile_graph
    from w2s.importers.builder import GraphBuilder
    from w2s.importers.onnx_import import load_onnx
"""

from w2s.core import (
    OpType,
    Operation,
    ComputeGraph,
    TensorWires,
    QuantConfig,
    QuantScheme,
    QuantGranularity,
)

__all__ = [
    "OpType", "Operation", "ComputeGraph", "TensorWires",
    "QuantConfig", "QuantScheme", "QuantGranularity",
]
