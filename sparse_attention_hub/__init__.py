"""Sparse Attention Hub (PQIS subset).

Minimal subset of the sparse-attention-hub research framework, containing
what is needed to run the PQ importance sampling (PQIS) masker on a benchmark.
"""

__version__ = "0.1.0"

from .adapters import (
    ModelAdapter,
    ModelAdapterHF,
    ModelHubAdapterInterface,
    Request,
    RequestResponse,
    SparseAttentionAdapterInterface,
)
from .metric_logging import MicroMetricLogger
from .sparse_attention import ResearchAttention, SparseAttention

__all__ = [
    "SparseAttention",
    "ResearchAttention",
    "MicroMetricLogger",
    "Request",
    "RequestResponse",
    "ModelHubAdapterInterface",
    "SparseAttentionAdapterInterface",
    "ModelAdapter",
    "ModelAdapterHF",
]
