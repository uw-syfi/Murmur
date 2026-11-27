"""Generate chunk-relative Whisper word timestamps for drafting and speaker attribution."""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock

import numpy as np

# A punctuated simplified-Chinese prompt makes Whisper write simplified characters
# and Chinese punctuation, matching the target models' Mandarin output.
_ZH_PROMPT = "以下是普通话的会议录音，使用简体中文和标点符号。好的，我们开始吧。"


def _word_converter(language):
    if language == "zh":
        try:
            import opencc
            return opencc.OpenCC("t2s").convert
        except ImportError:
            pass
    return str


class MLXWhisperDrafter:
    """Whisper on the Apple GPU, with word alignment from mlx-audio."""

    def __init__(self, language: str = "en"):
        import mlx.core as mx
        from mlx_audio.stt import load

        size = "small" if language == "zh" else "tiny"
        with mx.stream(mx.gpu):
            self.model = load(f"openai/whisper-{size}")
        self.language = language
        self.convert = _word_converter(language)
        self._lock = Lock()

    def __call__(self, wav16: np.ndarray) -> list[tuple[float, str]]:
        import mlx.core as mx

        if not len(wav16):
            return []
        # Drafting runs on a background thread. Use that thread's GPU stream.
        with self._lock, mx.stream(mx.gpu):
            result = self.model.generate(
                np.ascontiguousarray(wav16, dtype=np.float32), language=self.language,
                task="transcribe", temperature=0.0, word_timestamps=True,
                condition_on_previous_text=False, initial_prompt=_ZH_PROMPT if self.language == "zh" else None,
            )
        return [(float(w["start"]), self.convert(w["word"].strip()))
                for segment in result.segments for w in segment.get("words", []) if w["word"].strip()]


class WhisperDrafter:
    """faster-whisper with word timestamps (Whisper-tiny for English, Whisper-small for Mandarin)."""

    def __init__(self, language: str = "en", size: str | None = None, device: str = "cuda",
                 compute_type: str = "float16", batch_size: int = 16):
        from faster_whisper import BatchedInferencePipeline, WhisperModel
        size = size or ("small" if language == "zh" else "tiny")
        self.pipe = BatchedInferencePipeline(WhisperModel(size, device=device, compute_type=compute_type))
        self.language, self.batch_size = language, batch_size
        self.convert = _word_converter(language)

    def __call__(self, wav16: np.ndarray) -> list[tuple[float, str]]:
        """``(time_s, word)`` pairs for 16 kHz mono audio, times relative to its start."""
        kw = {"initial_prompt": _ZH_PROMPT} if self.language == "zh" else {}
        segs, _ = self.pipe.transcribe(np.ascontiguousarray(wav16, dtype=np.float32), language=self.language,
                                       task="transcribe", word_timestamps=True, batch_size=self.batch_size,
                                       beam_size=1, **kw)
        return [(float(w.start), self.convert(w.word.strip())) for s in segs for w in (s.words or []) if w.word.strip()]


def write_draft(draft_dir: Path, key: str, words: list[tuple[float, str]], language: str,
                chunk_offset: float = 0.0, diar_segments=None) -> None:
    """Write one chunk's draft file atomically (the server may be polling for it)."""
    draft_dir = Path(draft_dir)
    tmp = draft_dir / f".{key}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps({"language": language, "words": words, "chunk_offset": chunk_offset,
                               "diar_segments": diar_segments or []}))
    os.replace(tmp, draft_dir / f"{key}.json")
