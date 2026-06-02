"""KV-cache eviction policy tests — CPU-only, no model download.

These cover Murmur's core contribution: that each eviction policy bounds the
physical cache, stays rectangular under batching, and keeps ``get_mask_sizes()``
in lockstep with the tensor ``update()`` actually returns (the contract HF relies
on to build the attention mask — a mismatch silently corrupts attention).
"""
import pytest
import torch

from murmur.inference.kv_cache import (
    StreamingLayer,
    SpeechWindowLayer,
    SpeechStreamLayer,
    StreamingCache,
    SpeechWindowCache,
    SpeechStreamCache,
    make_cache,
    _validate_layer_ranges,
)


def _kv(B=1, H=2, S=1, D=4):
    return torch.randn(B, H, S, D)


def _positions(B, H, start, length, D=4):
    """K/V whose every feature equals the absolute position index, so we can
    assert exactly which positions survived an eviction."""
    pos = torch.arange(start, start + length, dtype=torch.float32)
    return pos.view(1, 1, length, 1).expand(B, H, length, D).contiguous()


# ── StreamingLayer ────────────────────────────────────────────────────────────

def test_streaming_prefill_never_evicted_and_budget_bounded():
    sink, window, prefill = 2, 4, 5
    layer = StreamingLayer(sink_size=sink, window_size=window, evict_block=1)

    layer.update(_kv(S=prefill), _kv(S=prefill))
    assert layer.keys.shape[-2] == prefill
    assert layer.prefill_len == prefill

    for i in range(30):
        cp = torch.tensor([prefill + i])
        predicted = layer.get_mask_sizes(cp)[0]
        k, _ = layer.update(_kv(S=1), _kv(S=1))
        assert k.shape[-2] == predicted, f"mask/update mismatch at step {i}"

    # prefill is preserved; decoded region capped at sink + window.
    assert layer.keys.shape[-2] == prefill + sink + window
    assert layer.get_seq_length() == prefill + 30  # logical, not physical


def test_streaming_keeps_correct_positions():
    sink, window, prefill = 2, 3, 4
    layer = StreamingLayer(sink_size=sink, window_size=window, evict_block=1)
    layer.update(_positions(1, 2, 0, prefill), _positions(1, 2, 0, prefill))

    n_decode = 12
    for i in range(n_decode):
        p = prefill + i
        layer.update(_positions(1, 2, p, 1), _positions(1, 2, p, 1))

    kept = layer.keys[0, 0, :, 0].tolist()
    decoded_first = list(range(prefill, prefill + sink))
    decoded_last = list(range(prefill + n_decode - window, prefill + n_decode))
    expected = list(range(prefill)) + decoded_first + decoded_last
    assert kept == [float(x) for x in expected]


def test_streaming_evict_block_reduces_firing_frequency():
    common = dict(sink_size=2, window_size=4)
    per_step = StreamingLayer(**common, evict_block=1)
    blocked = StreamingLayer(**common, evict_block=8)
    for layer in (per_step, blocked):
        layer.update(_kv(S=5), _kv(S=5))
        for _ in range(40):
            layer.update(_kv(S=1), _kv(S=1))
    assert blocked.eviction_count < per_step.eviction_count


# ── SpeechWindowLayer (batched) ─────────────────────────────────────────────────

def test_speech_window_batched_rectangular_and_capped():
    B, H, D, prefill = 2, 2, 4, 20
    layer = SpeechWindowLayer(
        speech_sink=2, speech_window=3, slide_delay=1, slide_rate=2,
        speech_evict_block=1,
    )
    # Per-row speech spans differ (left-padding); spans 12 and 13.
    layer.set_speech_region([3, 5], [15, 18])
    layer.update(_kv(B, H, prefill, D), _kv(B, H, prefill, D))

    for i in range(15):
        cp = torch.tensor([prefill + i])
        predicted = layer.get_mask_sizes(cp)[0]
        k, _ = layer.update(_kv(B, H, 1, D), _kv(B, H, 1, D))
        assert k.shape[0] == B, "K/V must stay rectangular across rows"
        assert k.shape[-2] == predicted, f"mask/update mismatch at step {i}"

    cap = min(12, 13) - 2 - 3  # min_span - sink - window
    assert layer.evicted == cap
    assert layer.keys.shape[-2] == prefill + 15 - cap


# ── SpeechStreamLayer (combined, batched) ───────────────────────────────────────

def test_speech_stream_batched_mask_contract():
    B, H, D, prefill = 2, 2, 4, 20
    layer = SpeechStreamLayer(
        speech_sink=2, speech_window=3, slide_delay=1, slide_rate=2,
        output_sink=1, output_window=3, output_evict_block=1, speech_evict_block=1,
    )
    layer.set_speech_region([3, 5], [15, 18])
    layer.update(_kv(B, H, prefill, D), _kv(B, H, prefill, D))

    for i in range(40):
        cp = torch.tensor([prefill + i])
        predicted = layer.get_mask_sizes(cp)[0]
        k, _ = layer.update(_kv(B, H, 1, D), _kv(B, H, 1, D))
        assert k.shape[0] == B
        assert k.shape[-2] == predicted, f"mask/update mismatch at step {i}"


# ── make_cache factory ──────────────────────────────────────────────────────────

def test_make_cache_returns_expected_types():
    assert make_cache(None) is None
    assert isinstance(make_cache("streaming"), StreamingCache)
    assert isinstance(make_cache("speech_window"), SpeechWindowCache)
    assert isinstance(make_cache("speech_stream"), SpeechStreamCache)


def test_make_cache_unknown_policy_raises():
    with pytest.raises(ValueError):
        make_cache("not_a_policy")


def test_cache_lazily_builds_one_layer_per_index_and_aggregates():
    cache = StreamingCache(sink_size=2, window_size=4)
    n_layers = 3
    for idx in range(n_layers):
        cache.update(_kv(S=5), _kv(S=5), layer_idx=idx)
    assert len(cache.layers) == n_layers
    phys, logi = cache.compression_stats
    assert logi == 5 * n_layers  # 5 prefill positions per layer
    assert phys == logi          # nothing evicted yet


# ── _validate_layer_ranges ──────────────────────────────────────────────────────

def test_validate_layer_ranges_accepts_sorted_disjoint():
    out = _validate_layer_ranges([{"start": 4, "end": 8}, {"start": 0, "end": 4}])
    assert [r["start"] for r in out] == [0, 4]  # returned sorted


@pytest.mark.parametrize("bad", [
    [{"start": 0, "end": 5}, {"start": 3, "end": 8}],  # overlap
    [{"start": -1, "end": 2}],                          # negative start
    [{"start": 2, "end": 2}],                           # end <= start
    [{"start": 0, "end": 4, "bogus": 1}],               # unknown key
])
def test_validate_layer_ranges_rejects_invalid(bad):
    with pytest.raises(ValueError):
        _validate_layer_ranges(bad)
