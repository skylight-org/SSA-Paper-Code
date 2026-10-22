"""Benchmark package (PQIS subset).

Only the benchmarks used for the PQIS evaluation are kept:
LongBench, RULER-32k, and a tiny in-memory MockBenchmark for smoke tests.
"""

from . import longbench, mock_benchmark, ruler32k
from .base import Benchmark
from .benchmark_registry import (
    create_benchmark_instance,
    ensure_benchmarks_loaded,
    get_available_benchmark_names,
    get_benchmark_subsets,
    get_registered_benchmarks,
    register_benchmark,
    validate_benchmark_config,
)
from .longbench import LongBench
from .mock_benchmark import MockBenchmark
from .ruler32k import Ruler32K

__all__ = [
    "Benchmark",
    "register_benchmark",
    "get_registered_benchmarks",
    "get_available_benchmark_names",
    "create_benchmark_instance",
    "ensure_benchmarks_loaded",
    "validate_benchmark_config",
    "get_benchmark_subsets",
    "LongBench",
    "Ruler32K",
    "MockBenchmark",
    "longbench",
    "ruler32k",
    "mock_benchmark",
]
