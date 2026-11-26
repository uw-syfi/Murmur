"""Exercise local MOSS loading and stitching on CPU with an in-memory decoder."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from murmur import Murmur
from murmur.models.mlx_moss import DEFAULT_CHECKPOINT


def test_mlx_transcribes_chunks_with_one_unquantized_model(monkeypatch):
    generate = Mock(return_value=SimpleNamespace(text="[0.00][S01]hello[0.50]"))
    loader = Mock(return_value=SimpleNamespace(generate=generate))
    monkeypatch.setitem(sys.modules, "moss_transcribe_diarize.mlx", SimpleNamespace(load_model=loader))

    with Murmur.load("moss", backend="mlx", diarize=False, chunk_s=1) as model:
        result = model.transcribe(np.zeros(32_000, dtype=np.float32), sr=16_000,
                                  timeline=[(0, 1, "A"), (1, 2, "B")])

    loader.assert_called_once_with(DEFAULT_CHECKPOINT, strict=True)
    assert generate.call_count == 2
    assert generate.call_args.args[0].shape == (16_000,)
    assert generate.call_args.kwargs == {"max_tokens": 512, "temperature": 0.0}
    assert result.stats["chunks_failed"] == 0
    assert [(s.start, s.end, s.speaker) for s in result.segments] == [(0, 0.5, "A"), (1, 1.5, "B")]
    assert model.client.model is None


@pytest.mark.parametrize("options", [{"model": "voxtral"}, {"url": "http://localhost:8000"},
                                     {"speculative": "ngram"}, {"batch_size": 2}])
def test_mlx_rejects_unsupported_settings_before_loading(options):
    with pytest.raises(ValueError, match="MLX"):
        Murmur.load(**({"model": "moss", "backend": "mlx"} | options))


@pytest.mark.parametrize("draft,budget", [([], 9), ([7, 7], 9), ([2, 3, 7], 9), ([2, 3, 4, 5], 9),
                                           ([2, 3, 4, 5], 3), ([], 1), ([], 0)])
def test_mlx_verifier_matches_greedy_after_rollback(draft, budget):
    from murmur.drafting.mlx_moss import verify_greedy

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
    from murmur.drafting import mlx_moss, whisper_drafts

    words = [(0.1, "hello")]
    whisper = Mock(return_value=words)
    whisper_loader = Mock(return_value=whisper)
    verify = Mock(return_value=("[0.00][S01]hello[0.50]", {"proposed": 4, "accepted": 2, "verification_calls": 3}))
    model = Mock()
    model.generate.return_value = SimpleNamespace(text="[0.00][S01]hello[0.50]")
    monkeypatch.setitem(sys.modules, "moss_transcribe_diarize.mlx", SimpleNamespace(load_model=Mock(return_value=model)))
    monkeypatch.setattr(whisper_drafts, "WhisperDrafter", whisper_loader)
    monkeypatch.setattr(mlx_moss, "generate", verify)
    audio = np.zeros(32_000, dtype=np.float32)
    timeline = [(0, 1, "A"), (1, 2, "B")]
    with Murmur.load("moss", backend="mlx", speculative="murmur", diarize=False, chunk_s=1) as pipeline:
        transcript = pipeline.transcribe(audio, sr=16_000, timeline=timeline)
        baseline = pipeline.transcribe(audio, sr=16_000, single_pass=True)
    whisper_loader.assert_called_once_with("en", device="cpu", compute_type="int8")
    assert whisper.call_count == 2
    assert [c.args[2:] for c in verify.call_args_list] == [
        (words, "en", [list(turn)], turn[0], 512) for turn in timeline
    ]
    assert transcript.stats["speculative"] == {"proposed": 8, "accepted": 4, "verification_calls": 6}
    assert baseline.stats["n_chunks"] == 1 and "speculative" not in baseline.stats
