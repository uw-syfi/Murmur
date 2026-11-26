"""Local model routing and decoding contracts without Metal or model downloads."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from murmur import Murmur
from murmur.drafting import whisper_drafts
from murmur.models.vibevoice import MLXVibeVoiceClient
from murmur.models.voxtral import MLXVoxtralClient


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


@pytest.mark.parametrize("name,repo", [("voxtral", "Voxtral-Mini-3B-2507"), ("voxtral-small", "Voxtral-Small-24B-2507")])
def test_voxtral_decodes_on_loading_thread_and_attributes_words(monkeypatch, name, repo):
    processor = Mock()
    processor.apply_transcrition_request.return_value = SimpleNamespace(input_ids=np.array([[0, 1]]), input_features=[])
    processor.decode.return_value = "hello"
    thread_ids = []

    def generate(**kwargs):
        thread_ids.append(threading.get_ident())
        assert kwargs["temperature"] == 0 and kwargs["repetition_penalty"] == 1
        return np.array([[0, 1, 77]])

    def load(client, checkpoint):
        thread_ids.append(threading.get_ident())
        assert checkpoint == f"efficient-speech/{repo}-MLX-INT4"
        client.processor = processor
        return SimpleNamespace(generate=generate)

    whisper = Mock(return_value=[(0.2, "hello")])
    monkeypatch.setattr(MLXVoxtralClient, "_load", load)
    monkeypatch.setattr(whisper_drafts, "WhisperDrafter", Mock(return_value=whisper))
    with Murmur.load(name, backend="mlx", quantization="int4", language="zh", diarize=False) as pipeline:
        result = pipeline.transcribe(np.zeros(16000, dtype=np.float32), sr=16000, timeline=[(0, 1, "A")])
    assert len(thread_ids) == 2 and thread_ids[0] == thread_ids[1]
    assert processor.apply_transcrition_request.call_args.kwargs == {"language": "zh", "sampling_rate": 16000}
    processor.decode.assert_called_once_with([77], skip_special_tokens=True)
    whisper.assert_called_once()
    assert result.text == "hello" and result.speakers == ["A"]
