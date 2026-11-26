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
    processor.apply_transcrition_request.return_value = SimpleNamespace(
        input_ids=np.array([[0, 1]]), input_features=np.zeros((1, 128, 3000), dtype=np.float32),
    )
    processor.decode.return_value = "hello"
    thread_ids = []

    def generate(**kwargs):
        thread_ids.append(threading.get_ident())
        assert kwargs["temperature"] == 0 and kwargs["repetition_penalty"] == 1
        assert kwargs["input_features"].dtype == np.float16
        return np.array([[0, 1, 77]])

    def load(client, checkpoint):
        thread_ids.append(threading.get_ident())
        assert checkpoint == f"efficient-speech/{repo}-MLX-INT4"
        client.processor = processor
        model = Mock(generate=generate)
        model.audio_tower.conv1.weight.dtype = np.float16
        return model

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


def test_vibevoice_routes_whisper_words_to_local_verifier(monkeypatch):
    from murmur.models import vibevoice

    model = Mock(is_streaming_model=False)
    model.tokenizer.encode.return_value = [1]
    words = [(0.1, "hello")]
    monkeypatch.setattr(MLXVibeVoiceClient, "_load", Mock(return_value=model))
    monkeypatch.setattr(whisper_drafts, "WhisperDrafter", Mock(return_value=Mock(return_value=words)))
    stats = dict(proposed=4, accepted=2, verification_calls=3)
    generate = Mock(return_value=('[{"Start":0,"End":0.5,"Speaker":0,"Content":"hello"}]', stats))
    monkeypatch.setattr(vibevoice, "_generate_with_draft", generate)
    with Murmur.load("vibevoice", backend="mlx", speculative="murmur", diarize=False) as pipeline:
        tr = pipeline.transcribe(np.zeros(16000, dtype=np.float32), sr=16000)
    assert tr.text == "hello" and tr.stats["speculative"] == stats
    assert generate.call_args.args[2].dt.words == ["hello"]


def test_vibevoice_batches_chunks_in_order_and_keeps_single_pass_serial(monkeypatch):
    from murmur.models.base import Reply

    model = Mock()
    model.generate.return_value.text = "[]"
    monkeypatch.setattr(MLXVibeVoiceClient, "_load", Mock(return_value=model))
    whisper = Mock(side_effect=[[(0., w)] for w in ("one", "two", "three")])
    monkeypatch.setattr(whisper_drafts, "WhisperDrafter", Mock(return_value=whisper))
    groups = []

    def batch(client, jobs, language):
        words = [draft[0][1] for _, draft in jobs]
        groups.append(words)
        return [Reply('[{"Start":0,"End":1,"Speaker":0,"Content":"' + w + '"}]', 0.1) for w in words]

    monkeypatch.setattr(MLXVibeVoiceClient, "decode_draft_batch", batch)
    wave = np.zeros(3 * 16000, dtype=np.float32)
    with Murmur.load("vibevoice", backend="mlx", batch_size=2, speculative="murmur",
                     chunk_s=1, diarize=False) as pipeline:
        result = pipeline.transcribe(wave, sr=16000)
        pipeline.transcribe(wave, sr=16000, single_pass=True)
    assert groups == [["one", "two"], ["three"]]
    assert result.text == "one two three"
    model.generate.assert_called_once()
