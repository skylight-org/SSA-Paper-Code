"""Sampling-based maskers.

Only the base class is kept: ResearchAttention needs it to enforce the
"at most one sampling masker" rule. PQImportance is deliberately a *fixed*
masker so that it can be stacked with a sampling masker.
"""

from .base import SamplingMasker, SamplingMaskerConfig

__all__ = [
    "SamplingMasker",
    "SamplingMaskerConfig",
]
