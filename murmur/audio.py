"""Audio loading, pyannote diarization, and target-chunk boundaries."""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger(__name__)
SR = 16_000

Timeline = list[tuple[float, float, str]]  # (start_s, end_s, speaker)


def load_audio(audio, sr: int | None = None) -> torch.Tensor:
    """16 kHz mono float32 waveform from a path, numpy array or tensor (give ``sr`` for arrays)."""
    import torchaudio
    if isinstance(audio, (str, Path)):
        import soundfile as sf
        data, sr = sf.read(str(audio), dtype="float32", always_2d=True)  # (frames, channels)
        wav = torch.from_numpy(data.mean(axis=1))
    else:
        wav = audio if isinstance(audio, torch.Tensor) else torch.as_tensor(np.asarray(audio, dtype=np.float32))
        if sr is None:
            raise ValueError("pass sr= with a raw waveform")
    wav = wav.float()
    if wav.ndim > 1:
        wav = wav.mean(0) if wav.shape[0] < wav.shape[-1] else wav.mean(-1)
    if sr != SR:
        wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, SR).squeeze(0)
    return wav.contiguous()


def resample(wav16: torch.Tensor, target_sr: int) -> np.ndarray:
    if target_sr == SR:
        return wav16.cpu().numpy().astype(np.float32)
    import torchaudio
    return torchaudio.functional.resample(wav16.unsqueeze(0), SR, target_sr).squeeze(0).cpu().numpy().astype(np.float32)


def hf_token(token: str | None = None) -> str | None:
    if token:
        return token
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    p = Path(os.path.expanduser("~/.cache/huggingface/token"))
    return p.read_text().strip() if p.exists() else None


class Diarizer:
    """pyannote speaker-diarization-3.1. Gives target-chunk boundaries and links speakers across chunks.

    Needs ``pyannote.audio`` and a Hugging Face token with access to the gated pyannote models.
    """

    def __init__(self, token: str | None = None, device: str | None = None):
        from pyannote.audio import Pipeline
        tok = hf_token(token)
        if not tok:
            raise RuntimeError("pyannote needs a Hugging Face token: set HF_TOKEN or run `huggingface-cli login`, "
                               "and accept the terms of pyannote/speaker-diarization-3.1.")
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=tok).to(dev)
        self._lock = threading.Lock()

    def __call__(self, wav16: torch.Tensor, min_speakers: int | None = None) -> Timeline:
        kw = {"min_speakers": min_speakers} if min_speakers else {}
        with self._lock:
            # cuDNN autotuning: without it the embedding model is ~4x slower on recent torch/cuDNN
            prev, torch.backends.cudnn.benchmark = torch.backends.cudnn.benchmark, True
            try:
                raw = self.pipe({"waveform": wav16.unsqueeze(0), "sample_rate": SR}, **kw)
            finally:
                torch.backends.cudnn.benchmark = prev
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # the vLLM server shares this GPU
        ann = raw if hasattr(raw, "itertracks") else (getattr(raw, "speaker_diarization", None)
                                                      or getattr(raw, "diarization", None)
                                                      or next(iter(vars(raw).values())))
        return [(float(t.start), float(t.end), str(spk)) for t, _, spk in ann.itertracks(yield_label=True)]


def chunk_bounds(duration_s: float, chunk_s: float, timeline: Timeline | None) -> list[tuple[float, float]]:
    """Target chunks of at most ``chunk_s`` seconds, cut between speech regions.

    Speech regions come from the diarization timeline (in its order); without one,
    the audio is cut into fixed ``chunk_s`` windows.
    """
    if not timeline:
        n = max(1, int(np.ceil(duration_s / chunk_s)))
        return [(i * chunk_s, min(duration_s, (i + 1) * chunk_s)) for i in range(n)]
    out = []
    start, end = timeline[0][0], timeline[0][1]
    for t0, t1, _ in timeline[1:]:
        if t1 - start <= chunk_s:
            end = t1
        else:
            out.append((start, end))
            start, end = t0, t1
    out.append((start, end))
    return out


def slice_audio(wav16: torch.Tensor, start: float, end: float, target_sr: int) -> np.ndarray:
    a, b = int(start * SR), min(int(end * SR), wav16.shape[0])
    return resample(wav16[a:b], target_sr)


class Timer:
    """Wall-clock stage timer: ``with t("vad"): ...``; ``t.stages`` maps name -> seconds."""

    def __init__(self):
        self.stages: dict[str, float] = {}

    def __call__(self, name):
        timer = self

        class _Ctx:
            def __enter__(self):
                self.t0 = time.perf_counter()

            def __exit__(self, *a):
                timer.stages[name] = timer.stages.get(name, 0.0) + time.perf_counter() - self.t0
        return _Ctx()
