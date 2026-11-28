"""Audio loading, diarization (pyannote or Nemotron), and target-chunk boundaries."""

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


def read_audio_file(file) -> tuple[np.ndarray, int]:
    """Read a path or an uploaded file stream as mono samples at its original rate."""
    import soundfile as sf
    data, sr = sf.read(file, dtype="float32", always_2d=True)
    return data.mean(axis=1), sr


def load_audio(audio, sr: int | None = None) -> torch.Tensor:
    """16 kHz mono float32 waveform from a path, numpy array or tensor (give ``sr`` for arrays)."""
    import torchaudio
    if isinstance(audio, (str, Path)):
        data, sr = read_audio_file(audio)
        wav = torch.from_numpy(data)
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

    min_speakers_hint = True  # takes ``min_speakers`` (the pipeline re-diarizes with the model's count)

    def __init__(self, token: str | None = None, device: str | None = None):
        from pyannote.audio import Pipeline
        tok = hf_token(token)
        if not tok:
            raise RuntimeError("pyannote needs a Hugging Face token: set HF_TOKEN or run `huggingface-cli login`, "
                               "and accept the terms of pyannote/speaker-diarization-3.1.")
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        self.pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=tok).to(self.device)
        self._lock = threading.Lock()

    def __call__(self, wav16: torch.Tensor, min_speakers: int | None = None, *,
                 num_speakers: int | None = None, hook=None) -> Timeline:
        kw = {"min_speakers": min_speakers} if min_speakers else {}
        if num_speakers is not None:
            kw["num_speakers"] = num_speakers
        if hook is not None:
            kw["hook"] = hook
        with self._lock:
            audio = {"waveform": wav16.unsqueeze(0), "sample_rate": SR}
            if self.device.type == "cuda":
                # cuDNN autotuning accelerates the embedding model on CUDA.
                prev, torch.backends.cudnn.benchmark = torch.backends.cudnn.benchmark, True
                try:
                    raw = self.pipe(audio, **kw)
                finally:
                    torch.backends.cudnn.benchmark = prev
                torch.cuda.empty_cache()  # the vLLM server shares this GPU
            else:
                raw = self.pipe(audio, **kw)
        ann = raw if hasattr(raw, "itertracks") else (getattr(raw, "speaker_diarization", None)
                                                      or getattr(raw, "diarization", None)
                                                      or next(iter(vars(raw).values())))
        return [(float(t.start), float(t.end), str(spk)) for t, _, spk in ann.itertracks(yield_label=True)]


PyannoteDiarizer = Diarizer


def merge_same_speaker(tl: Timeline, gap_s: float) -> Timeline:
    """Join each speaker's segments separated by less than ``gap_s`` (short pauses within a turn)."""
    out = []
    for spk in sorted({t[2] for t in tl}):
        cur = None
        for s, e, k in sorted(t for t in tl if t[2] == spk):
            if cur is not None and s - cur[1] < gap_s:
                cur[1] = max(cur[1], e)
            else:
                if cur is not None:
                    out.append(tuple(cur))
                cur = [s, e, k]
        if cur is not None:
            out.append(tuple(cur))
    return sorted(out)


class NemotronDiarizer:
    """NVIDIA Nemotron-3-Diarization (100M Sortformer, up to 8 speakers), a faster alternative to pyannote.

    Needs NeMo 3.1 or newer, which is not on PyPI yet (3.0.0 cannot load the model)::

        pip install "nemo-toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git"

    The model is not gated (OpenMDW-1.1 license), so no Hugging Face token is needed. Its
    segments are fine-grained, so each speaker's pauses shorter than ``merge_gap_s`` are joined;
    otherwise target-chunk cuts land inside utterances more often. It takes no speaker-count
    hint, so the pipeline does not re-diarize.
    """

    min_speakers_hint = False
    MODEL = "nvidia/Nemotron-3-Diarization"

    def __init__(self, device: str | None = None, merge_gap_s: float = 0.5, batch_size: int = 8):
        try:
            from nemo.collections.asr.models import SortformerEncLabelModel
        except ImportError as e:
            raise RuntimeError("diarizer='nemotron' needs NeMo 3.1+: pip install "
                               "\"nemo-toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git\"") from e
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        try:
            self.model = SortformerEncLabelModel.from_pretrained(self.MODEL).eval().to(dev)
        except Exception as e:
            if "rope" in str(e).lower():
                raise RuntimeError(f"{self.MODEL} needs NeMo 3.1+ (RoPE attention); PyPI's nemo-toolkit 3.0.0 "
                                   "cannot load it: pip install \"nemo-toolkit[asr] @ "
                                   "git+https://github.com/NVIDIA/NeMo.git\"") from e
            raise
        self.device, self.merge_gap_s, self.batch_size = dev, merge_gap_s, batch_size
        self._lock = threading.Lock()

    def __call__(self, wav16: torch.Tensor, min_speakers: int | None = None) -> Timeline:
        audio = [wav16.detach().cpu().numpy().astype(np.float32)]
        with self._lock, torch.inference_mode(), \
                torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            out = self.model.diarize(audio=audio, sample_rate=SR, batch_size=self.batch_size, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # the vLLM server shares this GPU
        lines = out[0] if out and isinstance(out[0], list) else out
        tl = sorted((float(p[0]), float(p[1]), str(p[2])) for p in (str(x).split() for x in lines) if len(p) == 3)
        return merge_same_speaker(tl, self.merge_gap_s) if self.merge_gap_s else tl


def make_diarizer(name: str = "pyannote", hf_token_: str | None = None, device: str | None = None):
    """A diarizer by name: "pyannote" (default) or "nemotron"."""
    if name == "pyannote":
        return Diarizer(hf_token_, device=device)
    if name == "nemotron":
        return NemotronDiarizer(device=device)
    raise ValueError(f"unknown diarizer {name!r}; choose 'pyannote' or 'nemotron'")

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


def max_length_bounds(duration_s: float, max_s: float | None, timeline: Timeline | None) -> list[tuple[float, float]]:
    """A "single pass" over audio longer than the model can take in one request: the
    fewest pieces of about equal length that each fit in ``max_s``, cut between speech
    regions (equal windows without a timeline). One piece when the audio fits."""
    if not max_s or duration_s <= max_s:
        return [(0.0, duration_s)]
    n = int(np.ceil(duration_s / max_s))
    if not timeline:
        step = duration_s / n
        return [(i * step, min(duration_s, (i + 1) * step)) for i in range(n)]
    return chunk_bounds(duration_s, min(max_s, 1.05 * duration_s / n), timeline)


def slice_audio(wav16: torch.Tensor, start: float, end: float, target_sr: int) -> np.ndarray:
    a, b = int(start * SR), min(int(end * SR), wav16.shape[0])
    return resample(wav16[a:b], target_sr)


def chunk_waveform(wav16: torch.Tensor, start: float, end: float, decode_sr: int) -> tuple[np.ndarray, np.ndarray]:
    """One slice of ``[start, end)`` resampled for the model and for 16 kHz draft/ASR.

    When ``decode_sr`` is 16 kHz, both arrays share the same buffer (MOSS, Voxtral).
    """
    a, b = int(start * SR), min(int(end * SR), wav16.shape[0])
    chunk = wav16[a:b]
    decode = resample(chunk, decode_sr)
    if decode_sr == SR:
        return decode, decode
    return decode, resample(chunk, SR)


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
