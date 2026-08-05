"""Prefill Cache Simulator."""

from .analyzer import TraceAnalysis, analyze_trace
from .identity import GeneratedBlockId, TraceBlockId
from .trace import TraceRecord, load_trace

__all__ = [
    "GeneratedBlockId",
    "TraceAnalysis",
    "TraceBlockId",
    "TraceRecord",
    "analyze_trace",
    "load_trace",
]
