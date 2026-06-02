"""Murmur — local chunked ASR inference + benchmarking for VibeVoice ASR."""

from .chunking import Chunk, get_chunks, get_chunks_from_timeline
from .inference import LocalBatchEngine, TranscriptSegment, InferenceStats
from .metrics import (
    MetricsResult,
    compute_wer,
    compute_cer,
    compute_cpwer,
    compute_tcpwer,
    compute_der,
    evaluate,
)

__version__ = "0.1.0"

__all__ = [
    # chunking
    "Chunk",
    "get_chunks",
    "get_chunks_from_timeline",
    # inference
    "LocalBatchEngine",
    "TranscriptSegment",
    "InferenceStats",
    # metrics
    "MetricsResult",
    "compute_wer",
    "compute_cer",
    "compute_cpwer",
    "compute_tcpwer",
    "compute_der",
    "evaluate",
]
