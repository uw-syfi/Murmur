"""Pyannote 3.1 diarization from a 16 kHz mono waveform to speaker turns."""

import json
import math
import threading
from pathlib import Path

import mlx.core as mx

from .clustering import assign, selected
from .networks import Segmentation, WeSpeaker

SAMPLE_RATE = 16000
WINDOW = 10 * SAMPLE_RATE
HOP = SAMPLE_RATE
FRAME_STEP = 270
FRAME_CENTER = 495.5


def timeline(activity, labels, speakers, duration, requested=None):
    chunks, frames, _ = activity.shape
    # Each 10 s window starts on a whole second. Round in integer sample
    # space to avoid accumulating floating-point drift on long recordings.
    divisor = math.gcd(HOP, FRAME_STEP)
    numerator, denominator = HOP // divisor, FRAME_STEP // divisor
    offsets = (mx.arange(chunks) * numerator + denominator // 2) // denominator
    positions = offsets[:, None] + mx.arange(frames)
    total = round(((chunks - 1) * HOP + WINDOW) / FRAME_STEP) + 1
    coverage = mx.zeros(total).at[positions].add(mx.ones((chunks, frames)))
    local_counts = activity.sum(axis=-1)
    counts = mx.round(mx.zeros(total).at[positions].add(local_counts) / mx.maximum(coverage, 1))
    if requested is not None:
        counts = mx.minimum(counts, requested)
    speakers = max(speakers, int(counts.max().item()))
    # Multiple local labels assigned to one speaker contribute only one vote
    # per window. Accumulate each global column independently to bound memory.
    votes = []
    for speaker in range(speakers):
        local = (activity * (labels == speaker)[:, None, :]).max(axis=-1)
        votes.append(mx.zeros(total).at[positions].add(local))
    ranks = mx.argsort(mx.argsort(-mx.stack(votes, axis=-1), axis=-1), axis=-1)
    active = ranks < counts[:, None]
    changes = mx.pad(active.astype(mx.int32), ((1, 1), (0, 0)))
    edges = changes[1:] - changes[:-1]
    turns = []
    for speaker in range(speakers):
        boundaries = selected(edges[:, speaker] != 0).tolist()
        for start, end in zip(boundaries[::2], boundaries[1::2]):
            t0 = (FRAME_CENTER + start * FRAME_STEP) / SAMPLE_RATE
            t1 = min(duration, (FRAME_CENTER + end * FRAME_STEP) / SAMPLE_RATE)
            if t1 > t0:
                turns.append((t0, t1, f"SPEAKER_{speaker:02d}"))
    return sorted(turns)


class MLXDiarizer:
    """Pyannote segmentation-3.0 and WeSpeaker ResNet34-LM, both on MLX."""
    segmentation_batch_size = 32
    embedding_batch_size = 8

    def __init__(self, token=None, *, checkpoint=None):
        if checkpoint is None:
            from huggingface_hub import snapshot_download
            checkpoint = snapshot_download(
                "efficient-speech/Pyannote-Diarization-3.1-MLX-FP32",
                revision="bbe1ad497e6eabc8096c6a47477e9c18dac0e65d",
                allow_patterns=["config.json", "segmentation.safetensors", "wespeaker.safetensors"], token=token,
            )
        directory = Path(checkpoint)
        config = json.loads((directory / "config.json").read_text())
        if config.get("format") != "murmur-diarization-mlx-v1":
            raise ValueError(f"Unsupported MLX diarization checkpoint in {directory}")
        with mx.stream(mx.gpu):
            self.segment = Segmentation(mx.load(str(directory / "segmentation.safetensors")))
            self.embed = WeSpeaker(mx.load(str(directory / "wespeaker.safetensors")))
        self._lock = threading.Lock()

    def __call__(self, wav16, min_speakers=None, *, num_speakers=None, hook=None):
        if any(count is not None and count < 1 for count in (min_speakers, num_speakers)):
            raise ValueError("Speaker counts must be positive.")
        if len(wav16) == 0:
            return []
        with self._lock, mx.stream(mx.gpu):
            if hasattr(wav16, "detach"):
                wav16 = wav16.detach().cpu().numpy()
            audio = mx.array(wav16, dtype=mx.float32)
            if audio.ndim != 1:
                raise ValueError("Expected a mono waveform.")
            duration = len(audio) / SAMPLE_RATE
            chunks = 1 + max(0, math.ceil((len(audio) - WINDOW) / HOP))
            padded = mx.pad(audio, (0, (chunks - 1) * HOP + WINDOW - len(audio)))
            windows = mx.as_strided(padded, (chunks, WINDOW), strides=(HOP, 1))
            activity = self._batches("segmentation", self.segment, self.segmentation_batch_size, hook, windows)
            if not activity.any().item():
                return []
            embeddings = self._batches("embeddings", self.embed, self.embedding_batch_size, hook, windows, activity)
            labels, speakers = assign(embeddings, activity, min_speakers, num_speakers)
            return timeline(activity, labels, speakers, duration, num_speakers)

    @staticmethod
    def _batches(name, model, size, hook, windows, *extra):
        batches = []
        if hook:
            hook(name, None, completed=0)
        for i in range(0, len(windows), size):
            # Materialize only this batch of overlapping windows.
            batch = mx.contiguous(windows[i:i + size])
            out = model(batch, *(arg[i:i + size] for arg in extra))
            mx.eval(out)
            batches.append(out)
        result = mx.concatenate(batches)
        mx.eval(result)
        if hook:
            hook(name, result)
        return result
