"""Nemotron-3 speaker diarization on the Apple GPU."""

import threading

import mlx.core as mx

from ..audio import SR, merge_same_speaker


class MLXNemotronDiarizer:
    MODEL = "mlx-community/Nemotron-3-Diarization"
    REVISION = "59ed2dbfc1346dcea9d423c71306a3a2499c568f"
    min_speakers_hint = False

    def __init__(self, merge_gap_s=0.5):
        try:
            from mlx_audio.vad import load
        except ImportError as exc:
            raise ImportError("Install the mlx-nemotron extra to run Nemotron diarization on MLX.") from exc

        with mx.stream(mx.gpu):
            self.model = load(self.MODEL, revision=self.REVISION, strict=True)
        self.merge_gap_s = merge_gap_s
        self._lock = threading.Lock()

    def __call__(self, wav16, min_speakers=None):
        if wav16.ndim != 1:
            raise ValueError("Expected a mono waveform.")
        if len(wav16) == 0:
            return []
        if hasattr(wav16, "detach"):
            wav16 = wav16.detach().cpu().numpy()
        with self._lock, mx.stream(mx.gpu):
            result = self.model.generate(wav16, sample_rate=SR)
        turns = [(s.start, s.end, f"SPEAKER_{s.speaker:02d}") for s in result.segments]
        return merge_same_speaker(turns, self.merge_gap_s)
