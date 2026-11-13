"""Draft windows: a short-window ASR (faster-whisper) transcribes each target chunk
in 30 s windows; its words, with times, become the text drafts the server-side
proposer reads."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

# A punctuated simplified-Chinese prompt makes Whisper write simplified characters
# and Chinese punctuation, matching the target models' Mandarin output.
_ZH_PROMPT = "以下是普通话的会议录音，使用简体中文和标点符号。好的，我们开始吧。"


class WhisperDrafter:
    """faster-whisper with word timestamps (Whisper-tiny for English, Whisper-small for Mandarin)."""

    def __init__(self, language: str = "en", size: str | None = None, device: str = "cuda",
                 compute_type: str = "float16", batch_size: int = 16):
        from faster_whisper import BatchedInferencePipeline, WhisperModel
        size = size or ("small" if language == "zh" else "tiny")
        self.pipe = BatchedInferencePipeline(WhisperModel(size, device=device, compute_type=compute_type))
        self.language, self.batch_size = language, batch_size
        self.t2s = None
        if language == "zh":  # Whisper may write traditional characters; the targets write simplified
            try:
                import opencc
                self.t2s = opencc.OpenCC("t2s").convert
            except ImportError:
                pass

    def __call__(self, wav16: np.ndarray) -> list[tuple[float, str]]:
        """``(time_s, word)`` pairs for 16 kHz mono audio, times relative to its start."""
        kw = {"initial_prompt": _ZH_PROMPT} if self.language == "zh" else {}
        segs, _ = self.pipe.transcribe(np.ascontiguousarray(wav16, dtype=np.float32), language=self.language,
                                       task="transcribe", word_timestamps=True, batch_size=self.batch_size,
                                       beam_size=1, **kw)
        conv = self.t2s or (lambda x: x)
        return [(float(w.start), conv(w.word.strip())) for s in segs for w in (s.words or []) if w.word.strip()]


def write_draft(draft_dir: Path, key: str, words: list[tuple[float, str]], language: str,
                chunk_offset: float = 0.0, diar_segments=None) -> None:
    """Write one chunk's draft file atomically (the server may be polling for it)."""
    draft_dir = Path(draft_dir)
    tmp = draft_dir / f".{key}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps({"language": language, "words": words, "chunk_offset": chunk_offset,
                               "diar_segments": diar_segments or []}))
    os.replace(tmp, draft_dir / f"{key}.json")
