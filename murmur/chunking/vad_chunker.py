from __future__ import annotations

import subprocess
from typing import NamedTuple

import numpy as np
import torch
import torchaudio

# ── Types ────────────────────────────────────────────────────────────────────

class Chunk(NamedTuple):
    start_s: float    # absolute start in the source recording (seconds)
    end_s: float      # absolute end in the source recording (seconds)
    wav_bytes: bytes  # ffmpeg-extracted WAV bytes at 16kHz mono


# ── Silero-VAD singleton ──────────────────────────────────────────────────────

_vad_model = None
_vad_utils = None


def load_vad():
    """Lazy-load Silero-VAD (cached after first call)."""
    global _vad_model, _vad_utils
    if _vad_model is None:
        _vad_model, _vad_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            verbose=False,
            force_reload=False,
            trust_repo=True,
        )
    return _vad_model, _vad_utils


# ── Audio loading ─────────────────────────────────────────────────────────────

def _load_16k_mono(audio_path: str) -> torch.Tensor:
    """Return a 16 kHz mono float32 1-D tensor for *audio_path*."""
    try:
        import torchaudio

        wav, sr = torchaudio.load(audio_path)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        return wav.squeeze(0)
    except Exception:
        import librosa  # fallback, but torch audio is preferred
        wav, _ = librosa.load(audio_path, sr=16000, mono=True)
        return torch.from_numpy(wav)


def _extract_wav_bytes(audio_path: str, start_s: float, end_s: float) -> bytes:
    """Extract [start_s, end_s] from *audio_path* via ffmpeg and return WAV bytes."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_s),
        "-to", str(end_s),
        "-i", audio_path,
        "-ar", "16000",
        "-ac", "1",
        "-f", "wav",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, check=True)
    return result.stdout


# ── Core chunking logic ───────────────────────────────────────────────────────

def _merge_to_chunks(
    segments: list[dict],
    max_chunk_s: float,
) -> list[tuple[float, float]]:
    """Greedily merge VAD speech segments into chunks ≤ max_chunk_s seconds.

    Each segment dict must have ``"start"`` and ``"end"`` keys (seconds).
    Single segments longer than *max_chunk_s* become their own chunk — they are
    not further split because the boundary decision belongs to the caller.
    """
    chunks: list[tuple[float, float]] = []
    chunk_start = segments[0]["start"]
    chunk_end = segments[0]["end"]

    for seg in segments[1:]:
        if seg["end"] - chunk_start <= max_chunk_s:
            chunk_end = seg["end"]
        else:
            chunks.append((chunk_start, chunk_end))
            chunk_start = seg["start"]
            chunk_end = seg["end"]

    chunks.append((chunk_start, chunk_end))
    return chunks


# ── Public API ────────────────────────────────────────────────────────────────
def get_chunks(
    audio_path: str,
    max_chunk_s: float = 30.0,
    min_silence_ms: int = 500,
    min_speech_ms: int = 250,
) -> list[Chunk]:
    """VAD-chunk an audio file.

    @Param
    audio_path:
        Path to any ffmpeg-decodable audio file.
    max_chunk_s:
        Maximum chunk duration in seconds (WhisperX default: 30 s)
    min_silence_ms:
        Minimum silence duration (ms) to split at.
    min_speech_ms:
        Minimum speech duration (ms) -> shorter regions are treated as noise.

    @Returns
    list[Chunk]
        Ordered list of non-overlapping chunks spanning all detected speech.
    """
    model, utils = load_vad()
    get_speech_ts = utils[0]

    wav = _load_16k_mono(audio_path)

    segments = get_speech_ts(
        wav,
        model,
        sampling_rate=16000,
        min_silence_duration_ms=min_silence_ms,
        min_speech_duration_ms=min_speech_ms,
        return_seconds=True,
    )

    if not segments:
        dur = wav.shape[0] / 16000.0
        return [Chunk(0.0, dur, _extract_wav_bytes(audio_path, 0.0, dur))]

    merged = _merge_to_chunks(segments, max_chunk_s)
    return [
        Chunk(start, end, _extract_wav_bytes(audio_path, start, end))
        for start, end in merged
    ]


def _slice_resample(
    wav: torch.Tensor,
    start_s: float,
    end_s: float,
    src_sr: int = 16_000,
    target_sr: int = 24_000,
) -> np.ndarray:
    start_i = int(start_s * src_sr)
    end_i = min(int(end_s * src_sr), wav.shape[0])
    chunk = wav[start_i:end_i]
    if src_sr != target_sr:
        chunk = torchaudio.functional.resample(chunk.unsqueeze(0), src_sr, target_sr).squeeze(0)
    return chunk.numpy()


def iter_chunks_tensor(
    wav: torch.Tensor,
    max_chunk_s: float = 30.0,
    min_silence_ms: int = 500,
    min_speech_ms: int = 250,
    diarization_timeline: list[tuple[float, float, str]] | None = None,
    src_sr: int = 16_000,
    target_sr: int = 24_000,
):
    if diarization_timeline:
        segments = [
            {"start": t0, "end": t1}
            for t0, t1, _ in diarization_timeline
        ]
    else:
        model, utils = load_vad()
        segments = utils[0](
            wav, model, sampling_rate=src_sr,
            min_silence_duration_ms=min_silence_ms,
            min_speech_duration_ms=min_speech_ms,
            return_seconds=True,
        )

    if not segments:
        dur = wav.shape[0] / src_sr
        yield 0.0, dur, _slice_resample(wav, 0.0, dur, src_sr, target_sr)
        return

    chunk_start = segments[0]["start"]
    chunk_end   = segments[0]["end"]

    for seg in segments[1:]:
        if seg["end"] - chunk_start <= max_chunk_s:
            chunk_end = seg["end"]
        else:
            yield chunk_start, chunk_end, _slice_resample(wav, chunk_start, chunk_end, src_sr, target_sr)
            chunk_start = seg["start"]
            chunk_end   = seg["end"]

    yield chunk_start, chunk_end, _slice_resample(wav, chunk_start, chunk_end, src_sr, target_sr)


def get_chunks_from_timeline(
    audio_path: str,
    timeline: list[tuple[float, float, str]],
    max_chunk_s: float = 30.0,
) -> list[Chunk]:
    """Chunk from a pyannote-style diarization timeline.

    @Param
    audio_path:
        Source audio file.
    timeline:
        List of ``(start_s, end_s, speaker_label)`` tuples from
        pyannote speaker diarization.
    max_chunk_s:
        Maximum chunk duration in seconds.
    """
    segments = [
        {"start": t0, "end": t1}
        for t0, t1, _ in timeline
    ]
    if not segments:
        return []
    merged = _merge_to_chunks(segments, max_chunk_s)
    return [
        Chunk(start, end, _extract_wav_bytes(audio_path, start, end))
        for start, end in merged
    ]
