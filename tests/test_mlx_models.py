"""Local model routing and decoding contracts without Metal or model downloads."""

from unittest.mock import Mock

import numpy as np

from murmur import Murmur
from murmur.models.vibevoice import MLXVibeVoiceClient


def test_vibevoice_uses_24khz_and_preserves_turns(monkeypatch):
    model = Mock()
    model.generate.return_value.text = '[{"Start":0.1,"End":0.5,"Speaker":2,"Content":"hello"}]'
    loader = Mock(return_value=model)
    monkeypatch.setattr(MLXVibeVoiceClient, "_load", loader)
    with Murmur.load("vibevoice", backend="mlx", quantization="int8", diarize=False) as pipeline:
        transcript = pipeline.transcribe(np.zeros(16000, dtype=np.float32), sr=16000)
    loader.assert_called_once_with("efficient-speech/VibeVoice-ASR-MLX-INT8")
    assert model.generate.call_args.args[0].shape == (24000,)
    assert [(s.speaker, s.start, s.end, s.text) for s in transcript.segments] == [("2", 0.1, 0.5, "hello")]

