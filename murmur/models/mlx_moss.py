"""Local MOSS inference using the community Apple Silicon MLX runtime."""

from pathlib import Path
from threading import Lock
from time import perf_counter

from .base import Reply
from .moss import MossModel

DEFAULT_CHECKPOINT = "vanch007/mlx-MOSS-Transcribe-Diarize"


class MLXMossClient(MossModel):
    def __init__(self, checkpoint: str | Path | None = None):
        try:
            from moss_transcribe_diarize.mlx import load_model
        except ImportError as exc:
            raise ImportError("Install the MOSS MLX runtime as described in the README.") from exc
        self.model = load_model(checkpoint or DEFAULT_CHECKPOINT, strict=True)
        self._lock = Lock()

    def decode(self, audio, request_id=None, retries=3) -> Reply:
        with self._lock:
            if self.model is None:
                raise RuntimeError("The MLX model is closed.")
            start = perf_counter()
            result = self.model.generate(audio, max_tokens=self.budget(audio), temperature=0.0)
            return Reply(result.text, perf_counter() - start)

    def decode_with_draft(self, audio, words, language, diar_segments, chunk_offset) -> Reply:
        from ..drafting.mlx_moss import generate

        with self._lock:
            if self.model is None:
                raise RuntimeError("The MLX model is closed.")
            start = perf_counter()
            text, stats = generate(self.model, audio, words, language, diar_segments, chunk_offset, self.budget(audio))
            return Reply(text, perf_counter() - start, speculative=stats)

    def close(self) -> None:
        with self._lock:
            self.model = None
