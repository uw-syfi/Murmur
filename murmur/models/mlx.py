"""Shared checkpoint selection and serial greedy decoding for local Apple MLX models."""

from pathlib import Path
from threading import Lock
from time import perf_counter

from .base import ModelClient, Reply


def published_checkpoint(hf_id: str, precision: str) -> str:
    if precision not in ("bf16", "int4", "int8"):
        raise ValueError(f"Unknown MLX quantization {precision!r}. Choose bf16, int4 or int8.")
    name = hf_id.split("/")[-1]
    return f"efficient-speech/{name}-MLX-{precision.upper()}"


class MLXClient(ModelClient):
    """Model-specific clients supply a loader and optionally a token-based generator."""

    def __init__(self, checkpoint: str | Path | None = None, quantization: str | None = None):
        if checkpoint is not None and quantization is not None:
            raise ValueError("An explicit MLX checkpoint selects its stored precision. Omit quantization.")
        self._lock = Lock()
        if checkpoint is None:
            precision = "bf16" if quantization is None else quantization
            checkpoint = published_checkpoint(self.hf_id, precision)
        self.model = self._load(str(checkpoint))

    def _load(self, checkpoint: str):
        raise NotImplementedError

    def _generate(self, audio, budget):
        result = self.model.generate(audio, max_tokens=budget, temperature=0.0)
        return result.text, None

    def decode(self, audio, request_id=None, retries=3) -> Reply:
        with self._lock:
            if self.model is None:
                raise RuntimeError("The MLX model is closed.")
            start = perf_counter()
            text, finish = self._generate(audio, self.budget(audio))
            return Reply(text, perf_counter() - start, finish)

    def close(self) -> None:
        with self._lock:
            self.model = None


def verify_greedy(first, proposer, verify, rewind, eos, budget):
    """Keep the last emitted token pending. The cache holds everything before it."""
    tokens = []
    stats = {"proposed": 0, "accepted": 0, "verification_calls": 0}
    if budget <= 0 or first in eos:
        return tokens, stats
    tokens.append(first)
    while len(tokens) < budget:
        # Leave room for the target's correction (or a bonus after a full match).
        k = min(8, budget - len(tokens) - 1)
        draft = proposer.propose(tokens, k)[:k] if k else []
        predictions = verify([tokens[-1], *draft])
        matched = 0
        while matched < len(draft) and predictions[matched] == draft[matched]:
            matched += 1
        stats["verification_calls"] += 1
        stats["proposed"] += len(draft)
        stats["accepted"] += matched
        if matched < len(draft):
            rewind(len(draft) - matched)
        for token in predictions[:matched + 1]:
            if token in eos:
                return tokens, stats
            tokens.append(token)
    return tokens, stats
