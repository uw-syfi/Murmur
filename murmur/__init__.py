"""Murmur: efficient long-context ASR inference with hierarchical chunking.

    from murmur import Murmur
    m = Murmur.load("vibevoice")
    print(m.transcribe("meeting.wav"))

Imports are lazy so that the vLLM server environment can import
``murmur.drafting.vllm_proposer`` without the client's dependencies.
"""

__version__ = "0.2.0"

__all__ = ["Murmur", "load", "Segment", "Transcript", "evaluate"]


def __getattr__(name):
    if name in ("Murmur", "load"):
        from . import api
        return getattr(api, name)
    if name in ("Segment", "Transcript"):
        from . import types
        return getattr(types, name)
    if name == "evaluate":
        from .metrics import evaluate
        return evaluate
    raise AttributeError(f"module 'murmur' has no attribute {name!r}")
