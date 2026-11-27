"""Exercise local MOSS loading and stitching on CPU with an in-memory decoder."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from murmur import Murmur


def test_mlx_transcription_loads_native_diarizer(monkeypatch):
    diarizer = Mock()
    monkeypatch.setitem(sys.modules, "murmur.diarization.mlx", SimpleNamespace(MLXDiarizer=diarizer))
    monkeypatch.setitem(sys.modules, "moss_transcribe_diarize.mlx", SimpleNamespace(load_model=Mock()))
    with Murmur.load("moss", backend="mlx", hf_token="test") as pipeline:
        assert pipeline.diarizer is diarizer.return_value
    diarizer.assert_called_once_with("test")


def test_mlx_transcribes_chunks_with_one_unquantized_model(monkeypatch):
    generate = Mock(return_value=SimpleNamespace(text="[0.00][S01]hello[0.50]"))
    loader = Mock(return_value=SimpleNamespace(generate=generate))
    monkeypatch.setitem(sys.modules, "moss_transcribe_diarize.mlx", SimpleNamespace(load_model=loader))

    with Murmur.load("moss", backend="mlx", diarize=False, chunk_s=1) as model:
        result = model.transcribe(np.zeros(32_000, dtype=np.float32), sr=16_000,
                                  timeline=[(0, 1, "A"), (1, 2, "B")])

    loader.assert_called_once_with("efficient-speech/MOSS-Transcribe-Diarize-MLX-BF16", strict=True)
    assert generate.call_count == 2
    assert generate.call_args.args[0].shape == (16_000,)
    assert generate.call_args.kwargs == {"max_tokens": 512, "temperature": 0.0}
    assert result.stats["chunks_failed"] == 0
    assert [(s.start, s.end, s.speaker) for s in result.segments] == [(0, 0.5, "A"), (1, 1.5, "B")]
    assert model.client.model is None


@pytest.mark.parametrize("options", [{"model": "voxtral", "draft_windows": True}, {"url": "http://localhost:8000"},
                                     {"speculative": "ngram"}, {"batch_size": 2}])
def test_mlx_rejects_unsupported_settings_before_loading(options):
    with pytest.raises(ValueError, match="MLX"):
        Murmur.load(**({"model": "moss", "backend": "mlx"} | options))


@pytest.mark.parametrize("draft,budget", [([], 9), ([7, 7], 9), ([2, 3, 7], 9), ([2, 3, 4, 5], 9),
                                           ([2, 3, 4, 5], 3), ([], 1), ([], 0)])
def test_mlx_verifier_matches_greedy_after_rollback(draft, budget):
    from murmur.models.moss import verify_greedy

    cache = []
    eos = 5

    def verify(block):
        # A stale rejected token changes every later prediction.
        predictions = []
        for token in block:
            cache.append(token)
            predictions.append(len(cache) + 1 if cache == list(range(1, len(cache) + 1)) else 99)
        return predictions

    def rewind(count):
        del cache[-count:]

    proposer = SimpleNamespace(propose=lambda tokens, k: draft[:k])
    output, stats = verify_greedy(1, proposer, verify, rewind, {eos}, budget)
    assert output == [1, 2, 3, 4][:budget]
    assert stats["accepted"] <= stats["proposed"]
    assert verify_greedy(eos, proposer, verify, rewind, {eos}, budget)[0] == []


def test_local_drafts_stay_with_their_chunk(monkeypatch):
    from murmur.drafting import whisper_drafts
    from murmur.models import moss

    words = [(0.1, "hello")]
    whisper = Mock(return_value=words)
    whisper_loader = Mock(return_value=whisper)
    verify = Mock(return_value=("[0.00][S01]hello[0.50]", {"proposed": 4, "accepted": 2, "verification_calls": 3}))
    model = Mock()
    model.generate.return_value = SimpleNamespace(text="[0.00][S01]hello[0.50]")
    monkeypatch.setitem(sys.modules, "moss_transcribe_diarize.mlx", SimpleNamespace(load_model=Mock(return_value=model)))
    monkeypatch.setattr(whisper_drafts, "MLXWhisperDrafter", whisper_loader)
    monkeypatch.setattr(moss, "_generate_with_draft", verify)
    audio = np.zeros(32_000, dtype=np.float32)
    timeline = [(0, 1, "A"), (1, 2, "B")]
    with Murmur.load("moss", backend="mlx", speculative="murmur", diarize=False, chunk_s=1) as pipeline:
        transcript = pipeline.transcribe(audio, sr=16_000, timeline=timeline)
        baseline = pipeline.transcribe(audio, sr=16_000, single_pass=True)
    whisper_loader.assert_called_once_with("en")
    assert whisper.call_count == 2
    assert [c.args[2:] for c in verify.call_args_list] == [
        (words, "en", [list(turn)], turn[0], 512) for turn in timeline
    ]
    assert transcript.stats["speculative"] == {"proposed": 8, "accepted": 4, "verification_calls": 6}
    assert baseline.stats["n_chunks"] == 1 and "speculative" not in baseline.stats


def test_mlx_quantization_uses_model_layer_selection(monkeypatch):
    from murmur.models.moss import MLXMossClient

    model = Mock()
    model.named_modules.return_value = []
    nn = SimpleNamespace(QuantizedLinear=type("Linear", (), {}), QuantizedEmbedding=type("Embedding", (), {}),
                         quantize=Mock())
    core = SimpleNamespace(eval=Mock())
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(nn=nn, core=core))
    monkeypatch.setitem(sys.modules, "mlx.nn", nn)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.setitem(sys.modules, "moss_transcribe_diarize.mlx", SimpleNamespace(load_model=Mock(return_value=model)))
    with Murmur.load("moss", backend="mlx", checkpoint="weights", quantization="int4", diarize=False):
        nn.quantize.assert_called_once_with(model, mode="affine", bits=4, group_size=64,
                                            class_predicate=model.model_quant_predicate)
        core.eval.assert_called_once_with(model.parameters.return_value)

    model.named_modules.return_value = [("layer", nn.QuantizedLinear())]
    with pytest.raises(ValueError, match="already quantized"):
        MLXMossClient("weights", "int4")


def test_reject_invalid_quantization_before_loading():
    with pytest.raises(ValueError, match="Unknown MLX quantization"):
        Murmur.load("moss", backend="mlx", quantization="int7")
    with pytest.raises(ValueError, match="MLX backend option"):
        Murmur.load("moss", backend="vllm", quantization="int4")


@pytest.mark.parametrize("bits", [4, 8])
def test_mlx_selects_published_quantized_checkpoint(monkeypatch, bits):
    loader = Mock()
    monkeypatch.setitem(sys.modules, "moss_transcribe_diarize.mlx", SimpleNamespace(load_model=loader))
    with Murmur.load("moss", backend="mlx", quantization=f"int{bits}", diarize=False):
        loader.assert_called_once_with(f"efficient-speech/MOSS-Transcribe-Diarize-MLX-INT{bits}", strict=True)
        loader.return_value.named_modules.assert_not_called()
