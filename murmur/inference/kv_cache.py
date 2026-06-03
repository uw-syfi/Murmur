"""KV cache eviction policies for VibeVoice ASR.

Plugs into HF transformers via ``past_key_values=`` on ``model.generate``.
Targets the transformers >= 4.50 ``Cache`` / ``CacheLayerMixin`` API where
the ``Cache`` is a container of per-layer ``CacheLayerMixin`` objects and
the eviction logic lives in the layer's ``update()``.

A fresh cache is instantiated per batch (they are stateful).
"""
from __future__ import annotations

from typing import Any, Optional

import torch
from transformers import Cache
from transformers.cache_utils import CacheLayerMixin


class StreamingLayer(CacheLayerMixin):
    """Attention-sink + sliding-window eviction (StreamingLLM-style), per layer.

    Prefill is stored in full and never evicted (it holds the audio context).
    Once the decoded region exceeds ``sink_size + window_size``, only the first
    ``sink_size`` and the last ``window_size`` decoded tokens are kept; the
    middle is dropped. RoPE positions of surviving K/V are preserved.
    """

    is_sliding = False  # not a uniform sliding window — prefill is protected

    def __init__(self, sink_size: int = 4, window_size: int = 1024,
                 evict_block: int = 1):
        super().__init__()
        self.sink_size = sink_size
        self.window_size = window_size
        # Trim back to budget only once the region overshoots by evict_block, so
        # the full-cache rebuild fires once per block steps instead of every step.
        self.evict_block = max(1, evict_block)
        self.cumulative_length = 0
        self.prefill_len: Optional[int] = None
        self.eviction_count = 0
        self.is_initialized = False

    def lazy_initialization(self, key_states: torch.Tensor) -> None:
        self.dtype, self.device = key_states.dtype, key_states.device
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states)

        self.cumulative_length += key_states.shape[-2]

        full_keys = torch.cat([self.keys, key_states], dim=-2)
        full_values = torch.cat([self.values, value_states], dim=-2)

        if self.prefill_len is None:
            # Prefill: keep everything.
            self.prefill_len = key_states.shape[-2]
            self.keys = full_keys
            self.values = full_values
            return self.keys, self.values

        # Decode: shrink storage if the post-prefill region is over budget.
        pre = self.prefill_len
        total = full_keys.shape[-2]
        post = total - pre
        budget = self.sink_size + self.window_size

        if post > budget + self.evict_block - 1:
            self.keys = torch.cat([
                full_keys[..., : pre + self.sink_size, :],
                full_keys[..., -self.window_size :, :],
            ], dim=-2)
            self.values = torch.cat([
                full_values[..., : pre + self.sink_size, :],
                full_values[..., -self.window_size :, :],
            ], dim=-2)
            self.eviction_count += 1
        else:
            self.keys = full_keys
            self.values = full_values

        # Return the post-eviction tensors so attention shapes match
        # get_mask_sizes() and the externally-built attention_mask.
        return self.keys, self.values

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        # Mask size must match the physical K/V size update() returns after any
        # eviction this step, so predict that size (mirroring update()'s logic).
        query_length = cache_position.shape[0]
        if self.prefill_len is None:
            return query_length, 0
        prior_storage = self.keys.shape[-2] if self.is_initialized else 0
        total = prior_storage + query_length
        budget = self.sink_size + self.window_size
        post = total - self.prefill_len
        if post > budget + self.evict_block - 1:
            kv_length = self.prefill_len + budget
        else:
            kv_length = total
        return kv_length, 0

    def get_seq_length(self) -> int:
        # Logical position (total tokens seen), not physical storage. HF slices
        # input_ids by this in prepare_inputs_for_generation, so it must reflect
        # full history, not the trimmed cache.
        return self.cumulative_length

    def get_max_cache_shape(self) -> int:
        return -1


class StreamingCache(Cache):
    """Container holding one ``StreamingLayer`` per LM layer.

    ``layer_class_to_replicate`` can't carry constructor args, so we keep
    ``layers=[]`` and override ``update()`` to lazily append our own
    parameterized layers as new ``layer_idx`` values appear.
    """

    def __init__(self, sink_size: int = 4, window_size: int = 1024,
                 evict_block: int = 1):
        super().__init__(layers=[])
        self.sink_size = sink_size
        self.window_size = window_size
        self.evict_block = max(1, evict_block)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        while len(self.layers) <= layer_idx:
            self.layers.append(
                StreamingLayer(self.sink_size, self.window_size, self.evict_block)
            )
        return self.layers[layer_idx].update(key_states, value_states, cache_kwargs)

    @property
    def eviction_count(self) -> int:
        """Total eviction firings across all layers (sum, not max)."""
        return sum(getattr(l, "eviction_count", 0) for l in self.layers)

    @property
    def compression_stats(self) -> tuple[int, int]:
        """(physical_kv, logical_kv) summed across layers.

        physical = K/V positions actually stored now; logical = positions an
        unbounded cache would hold. physical/logical is the fraction of KV kept.
        """
        physical = sum(
            (l.keys.shape[-2] if getattr(l, "is_initialized", False) else 0)
            for l in self.layers
        )
        logical = sum(getattr(l, "cumulative_length", 0) for l in self.layers)
        return physical, logical


class SpeechWindowLayer(CacheLayerMixin):
    """Sink + sliding-window eviction over the speech-token region of the
    prefill. Decoded tokens are never evicted by this policy.

    Schedule, per decode step ``t``:
      - t <= slide_delay: nothing evicted
      - t  > slide_delay: drop ``min(slide_rate * (t - slide_delay), cap)``
        speech tokens starting at ``speech_start + speech_sink``, where
        ``cap = speech_span - speech_sink - speech_window``.

    Batched (batch_size > 1): the per-step drop count depends only on ``t``, so
    it is uniform across rows and the K/V stays rectangular. Only the drop
    position differs per row (left-padding shifts each speech span), so eviction
    uses a per-row ``torch.gather``. The cap uses the minimum span in the batch
    so no row evicts into its own sink/window.
    """

    is_sliding = False

    def __init__(
        self,
        speech_sink: int = 32,
        speech_window: int = 512,
        slide_delay: int = 32,
        slide_rate: int = 8,
        speech_evict_block: int = 1,
    ):
        super().__init__()
        self.speech_sink = speech_sink
        self.speech_window = speech_window
        self.slide_delay = slide_delay
        self.slide_rate = slide_rate
        # Fire the speech gather once per ``block`` steps (dropping a block of
        # tokens) instead of every step. With ``slide_rate=1`` the un-blocked
        # schedule drops 1 token/step → a full-cache gather on *every* decode
        # step until the cap; blocking cuts that firing frequency by ~block×.
        self.speech_evict_block = max(1, speech_evict_block)
        self.cumulative_length = 0
        self.prefill_len: Optional[int] = None
        # Per-row speech-region boundaries (length-B lists). For batch_size==1
        # these are single-element. ``speech_span_min`` drives the (uniform)
        # eviction cap so every row drops the same count → rectangular K/V.
        self.speech_start_list: list[int] = []
        self.speech_end_list: list[int] = []
        self.speech_span_min: Optional[int] = None
        self.decode_step = 0
        self.evicted = 0
        self.eviction_count = 0
        self.is_initialized = False
        self._sds_cache: Optional[torch.Tensor] = None  # cached drop starts

    def set_speech_region(self, start, end) -> None:
        """``start``/``end`` may be scalars (batch_size==1) or per-row
        sequences (lists / 1-D tensors) of length B for batched eviction."""
        def _to_list(x) -> list[int]:
            if torch.is_tensor(x):
                return [int(v) for v in x.tolist()]
            if isinstance(x, (list, tuple)):
                return [int(v) for v in x]
            return [int(x)]
        self.speech_start_list = _to_list(start)
        self.speech_end_list = _to_list(end)
        self.speech_span_min = min(
            e - s for s, e in zip(self.speech_start_list, self.speech_end_list)
        )
        self._sds_cache = None

    def lazy_initialization(self, key_states: torch.Tensor) -> None:
        self.dtype, self.device = key_states.dtype, key_states.device
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def _speech_drop_starts(self, device) -> torch.Tensor:
        """Per-row physical drop start (speech_start_b + speech_sink), cached as
        a device tensor so it isn't rebuilt from a Python list each step."""
        if self._sds_cache is None or self._sds_cache.device != device:
            starts = torch.tensor(
                self.speech_start_list, device=device, dtype=torch.long
            )
            self._sds_cache = starts + self.speech_sink
        return self._sds_cache

    def _target_evict(self, step: int) -> int:
        if self.speech_span_min is None:
            return 0
        # Cap on the minimum span in the batch so no row evicts into its window.
        cap = max(0, self.speech_span_min - self.speech_sink - self.speech_window)
        if step <= self.slide_delay:
            return 0
        target = min(self.slide_rate * (step - self.slide_delay), cap)
        # Snap to a multiple of the block so the gather fires once per block.
        if self.speech_evict_block > 1:
            target = (target // self.speech_evict_block) * self.speech_evict_block
        return target

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states)

        self.cumulative_length += key_states.shape[-2]

        full_keys = torch.cat([self.keys, key_states], dim=-2)
        full_values = torch.cat([self.values, value_states], dim=-2)

        if self.prefill_len is None:
            self.prefill_len = key_states.shape[-2]
            self.keys = full_keys
            self.values = full_values
            return self.keys, self.values

        self.decode_step += 1
        target = self._target_evict(self.decode_step)
        new_drops = target - self.evicted

        if new_drops > 0:
            # Drop the next `new_drops` tokens at each row's physical offset
            # (speech_start_b + speech_sink). The count is uniform across rows,
            # so the result stays rectangular; only the offset differs, so kept
            # columns are selected with a per-row gather built arithmetically
            # (no boolean indexing, which would force a host<->device sync).
            B, H = full_keys.shape[0], full_keys.shape[1]
            S, D = full_keys.shape[-2], full_keys.shape[-1]
            device = full_keys.device
            keep = S - new_drops
            j = torch.arange(keep, device=device)[None, :]  # [1, keep]
            drop_start = self._speech_drop_starts(device)[:, None]  # [B, 1]
            src = j.expand(B, keep) + new_drops * (j >= drop_start).long()
            idx = src[:, None, :, None].expand(B, H, keep, D)
            self.keys = torch.gather(full_keys, 2, idx)
            self.values = torch.gather(full_values, 2, idx)
            self.evicted = target
            self.eviction_count += 1
        else:
            self.keys = full_keys
            self.values = full_values

        return self.keys, self.values

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        query_length = cache_position.shape[0]
        if self.prefill_len is None:
            return query_length, 0
        prior_storage = self.keys.shape[-2] if self.is_initialized else 0
        # Predict eviction that will fire on the *upcoming* update() call.
        next_step = self.decode_step + 1
        target = self._target_evict(next_step)
        new_drops = max(0, target - self.evicted)
        return prior_storage + query_length - new_drops, 0

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def get_max_cache_shape(self) -> int:
        return -1


_LAYER_RANGE_KEYS = {
    "start", "end", "speech_sink", "speech_window", "slide_delay", "slide_rate",
}


def _validate_layer_ranges(ranges: list[dict]) -> list[dict]:
    """Normalize, sort, and overlap-check a list of per-layer range overrides.

    Each range is ``{"start": int, "end": int|None, ...overrides}`` covering
    layers ``[start, end)``. ``end=None`` means "to the last layer". Ranges
    must not overlap; gaps fall back to scalar defaults.
    """
    cleaned: list[dict] = []
    for i, r in enumerate(ranges):
        unknown = set(r) - _LAYER_RANGE_KEYS
        if unknown:
            raise ValueError(
                f"layer_ranges[{i}]: unknown keys {sorted(unknown)}; "
                f"allowed: {sorted(_LAYER_RANGE_KEYS)}"
            )
        start, end = r.get("start"), r.get("end")
        if not isinstance(start, int) or start < 0:
            raise ValueError(
                f"layer_ranges[{i}]: 'start' must be a non-negative int, got {start!r}"
            )
        if end is not None and (not isinstance(end, int) or end <= start):
            raise ValueError(
                f"layer_ranges[{i}]: 'end' must be null or int > start ({start}), got {end!r}"
            )
        cleaned.append(dict(r))
    cleaned.sort(key=lambda r: r["start"])
    for a, b in zip(cleaned, cleaned[1:]):
        a_end = a["end"]
        if a_end is None or a_end > b["start"]:
            raise ValueError(
                f"layer_ranges overlap: [{a['start']}, {a_end}) overlaps "
                f"[{b['start']}, {b['end']})"
            )
    return cleaned


class SpeechWindowCache(Cache):
    """Container of one ``SpeechWindowLayer`` per LM layer.

    ``layer_ranges`` optionally overrides the scalar defaults for specific
    layer-index ranges (half-open ``[start, end)``; ``end=None`` means "to
    last layer"). Layers not covered fall back to the scalar args, letting
    different layers use different eviction aggressiveness.
    """

    def __init__(
        self,
        speech_sink: int = 32,
        speech_window: int = 512,
        slide_delay: int = 32,
        slide_rate: int = 8,
        speech_evict_block: int = 1,
        layer_ranges: Optional[list[dict]] = None,
    ):
        super().__init__(layers=[])
        self.speech_sink = speech_sink
        self.speech_window = speech_window
        self.slide_delay = slide_delay
        self.slide_rate = slide_rate
        self.speech_evict_block = max(1, speech_evict_block)
        self.layer_ranges = _validate_layer_ranges(layer_ranges or [])
        # (start, end) scalars or per-row sequences; replayed onto new layers.
        self._speech_region: Optional[tuple[Any, Any]] = None

    def _params_for_layer(self, layer_idx: int) -> tuple[int, int, int, int]:
        for rng in self.layer_ranges:
            end = rng["end"]
            if rng["start"] <= layer_idx and (end is None or layer_idx < end):
                return (
                    int(rng.get("speech_sink", self.speech_sink)),
                    int(rng.get("speech_window", self.speech_window)),
                    int(rng.get("slide_delay", self.slide_delay)),
                    int(rng.get("slide_rate", self.slide_rate)),
                )
        return (self.speech_sink, self.speech_window, self.slide_delay, self.slide_rate)

    def set_speech_region(self, start, end) -> None:
        """``start``/``end`` are scalars (batch_size==1) or per-row sequences
        (lists / 1-D tensors of length B) for batched eviction. Stored verbatim
        so lazily-created layers receive the same per-row boundaries."""
        self._speech_region = (start, end)
        for layer in self.layers:
            layer.set_speech_region(start, end)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        while len(self.layers) <= layer_idx:
            sink, window, delay, rate = self._params_for_layer(len(self.layers))
            layer = SpeechWindowLayer(
                sink, window, delay, rate, self.speech_evict_block
            )
            if self._speech_region is not None:
                layer.set_speech_region(*self._speech_region)
            self.layers.append(layer)
        return self.layers[layer_idx].update(key_states, value_states, cache_kwargs)

    @property
    def eviction_count(self) -> int:
        return sum(getattr(l, "eviction_count", 0) for l in self.layers)

    @property
    def compression_stats(self) -> tuple[int, int]:
        """(physical_kv, logical_kv) summed across layers. See StreamingCache."""
        physical = sum(
            (l.keys.shape[-2] if getattr(l, "is_initialized", False) else 0)
            for l in self.layers
        )
        logical = sum(getattr(l, "cumulative_length", 0) for l in self.layers)
        return physical, logical


class SpeechStreamLayer(CacheLayerMixin):
    """Combined eviction: speech-region eviction over the prefill AND
    StreamingLLM sink+window over the decoded (output) tokens, in one layer.

    The two regions are disjoint (speech is in the prefill, generated tokens
    come after), so each decode step drops, per row:
      - speech: a chunk at ``speech_start_b + speech_sink`` (per-row offset,
        same schedule as :class:`SpeechWindowLayer`)
      - output: a chunk at ``prefill_phys + output_sink``, keeping the first
        ``output_sink`` and last ``output_window`` decoded tokens

    Both drop counts depend only on ``decode_step``, so they are uniform across
    rows and the K/V stays rectangular (batched-safe via a single gather).
    """

    is_sliding = False

    def __init__(
        self,
        speech_sink: int = 32,
        speech_window: int = 512,
        slide_delay: int = 32,
        slide_rate: int = 8,
        output_sink: int = 8,
        output_window: int = 512,
        output_evict_block: int = 1,
        speech_evict_block: int = 1,
        verbose: bool = False,
    ):
        super().__init__()
        self.speech_sink = speech_sink
        self.speech_window = speech_window
        self.slide_delay = slide_delay
        self.slide_rate = slide_rate
        self.output_sink = output_sink
        self.output_window = output_window
        self.output_evict_block = max(1, output_evict_block)
        self.speech_evict_block = max(1, speech_evict_block)
        self.cumulative_length = 0
        self.prefill_len: Optional[int] = None
        self.speech_start_list: list[int] = []
        self.speech_end_list: list[int] = []
        self.speech_span_min: Optional[int] = None
        self.decode_step = 0
        self.speech_evicted = 0
        self.output_evicted = 0
        self.eviction_count = 0
        self.is_initialized = False
        self._sds_cache: Optional[torch.Tensor] = None  # cached drop starts
        self.verbose = verbose
        self._eviction_log: list[dict] = []

    def set_speech_region(self, start, end) -> None:
        def _to_list(x) -> list[int]:
            if torch.is_tensor(x):
                return [int(v) for v in x.tolist()]
            if isinstance(x, (list, tuple)):
                return [int(v) for v in x]
            return [int(x)]
        self.speech_start_list = _to_list(start)
        self.speech_end_list = _to_list(end)
        self.speech_span_min = min(
            e - s for s, e in zip(self.speech_start_list, self.speech_end_list)
        )
        self._sds_cache = None

    def lazy_initialization(self, key_states: torch.Tensor) -> None:
        self.dtype, self.device = key_states.dtype, key_states.device
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def _speech_drop_starts(self, device) -> torch.Tensor:
        """Per-row physical drop start (speech_start_b + speech_sink), cached as
        a device tensor so we don't rebuild it from a Python list each step."""
        if self._sds_cache is None or self._sds_cache.device != device:
            starts = torch.tensor(
                self.speech_start_list, device=device, dtype=torch.long
            )
            self._sds_cache = starts + self.speech_sink
        return self._sds_cache

    def _speech_target(self, step: int) -> int:
        if self.speech_span_min is None:
            return 0
        cap = max(0, self.speech_span_min - self.speech_sink - self.speech_window)
        if step <= self.slide_delay:
            return 0
        target = min(self.slide_rate * (step - self.slide_delay), cap)
        if self.speech_evict_block > 1:
            target = (target // self.speech_evict_block) * self.speech_evict_block
        return target

    def _output_target(self, decoded_count: int) -> int:
        raw = max(0, decoded_count - self.output_sink - self.output_window)
        # Snap to a multiple of the block so the gather fires once per block
        # (the window overshoots by up to block-1 tokens between firings).
        if self.output_evict_block > 1:
            raw = (raw // self.output_evict_block) * self.output_evict_block
        return raw

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states)

        self.cumulative_length += key_states.shape[-2]

        full_keys = torch.cat([self.keys, key_states], dim=-2)
        full_values = torch.cat([self.values, value_states], dim=-2)

        if self.prefill_len is None:
            self.prefill_len = key_states.shape[-2]
            self.keys = full_keys
            self.values = full_values
            return self.keys, self.values

        self.decode_step += 1
        speech_target = self._speech_target(self.decode_step)
        speech_new = speech_target - self.speech_evicted
        # One decoded token is produced per decode step.
        output_target = self._output_target(self.decode_step)
        output_new = output_target - self.output_evicted

        if speech_new <= 0 and output_new <= 0:
            self.keys = full_keys
            self.values = full_values
            return self.keys, self.values

        S = full_keys.shape[-2]
        if speech_new <= 0:
            # Output-only firing (steady-state decode tail). The output drop
            # offset is uniform across rows, so the kept set is two contiguous
            # runs: [0, ods) and [ods + output_new, S). A slice+cat copies whole
            # head-dim vectors and is faster here than a gather, which reads
            # non-contiguously when the index is constant along D.
            prefill_phys = self.prefill_len - self.speech_evicted
            ods = prefill_phys + self.output_sink  # uniform output drop start
            self.keys = torch.cat(
                [full_keys[..., :ods, :], full_keys[..., ods + output_new:, :]],
                dim=-2,
            )
            self.values = torch.cat(
                [full_values[..., :ods, :], full_values[..., ods + output_new:, :]],
                dim=-2,
            )
        else:
            # Speech is actively sliding (possibly alongside output): the speech
            # hole sits at a different physical offset per row under left-padding,
            # so this phase needs a gather. Source indices are built
            # arithmetically (no boolean indexing) to avoid a host<->device sync.
            # The speech hole precedes the output hole for every row, so kept
            # slot j maps to j shifted right past each hole it has passed.
            B, H = full_keys.shape[0], full_keys.shape[1]
            D = full_keys.shape[-1]
            device = full_keys.device
            keep = S - speech_new - output_new  # uniform per row → rectangular
            j = torch.arange(keep, device=device)[None, :]  # [1, keep]
            sds = self._speech_drop_starts(device)[:, None]  # [B, 1]
            prefill_phys = self.prefill_len - self.speech_evicted
            ods = prefill_phys + self.output_sink  # uniform output drop start
            src = j.expand(B, keep).clone()
            src = src + speech_new * (j >= sds).long()
            if output_new > 0:
                # In post-speech-shift coords the output hole begins at
                # ods - speech_new.
                src = src + output_new * (j >= (ods - speech_new)).long()
            idx = src[:, None, :, None].expand(B, H, keep, D)
            self.keys = torch.gather(full_keys, 2, idx)
            self.values = torch.gather(full_values, 2, idx)

        self.speech_evicted = speech_target
        self.output_evicted = output_target
        self.eviction_count += 1

        if self.verbose and self.eviction_count <= 20:  # Log first 20 firings
            self._eviction_log.append({
                "step": self.decode_step,
                "speech_new": speech_new,
                "output_new": output_new,
                "kv_before": S,
                "kv_after": S - speech_new - output_new,
                "both_active": speech_new > 0 and output_new > 0,
            })

        return self.keys, self.values

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        query_length = cache_position.shape[0]
        if self.prefill_len is None:
            return query_length, 0
        prior_storage = self.keys.shape[-2] if self.is_initialized else 0
        next_step = self.decode_step + 1
        speech_new = max(0, self._speech_target(next_step) - self.speech_evicted)
        output_new = max(0, self._output_target(next_step) - self.output_evicted)
        return prior_storage + query_length - speech_new - output_new, 0

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def get_max_cache_shape(self) -> int:
        return -1

    def eviction_summary(self) -> str:
        """Return a human-readable summary of eviction events."""
        if not self._eviction_log:
            return "No evictions fired."
        both_count = sum(1 for e in self._eviction_log if e["both_active"])
        speech_only = sum(1 for e in self._eviction_log if e["speech_new"] > 0 and e["output_new"] == 0)
        output_only = sum(1 for e in self._eviction_log if e["speech_new"] == 0 and e["output_new"] > 0)
        lines = [f"Evictions: {len(self._eviction_log)} total"]
        lines.append(f"  Both speech+output active: {both_count}")
        lines.append(f"  Speech only: {speech_only}")
        lines.append(f"  Output only: {output_only}")
        if self._eviction_log:
            first = self._eviction_log[0]
            lines.append(f"  First: step {first['step']} (speech_new={first['speech_new']}, output_new={first['output_new']})")
        return "\n".join(lines)


class SpeechStreamCache(Cache):
    """Container of one :class:`SpeechStreamLayer` per LM layer — combined
    speech-region + decoded-token (StreamingLLM) eviction."""

    def __init__(
        self,
        speech_sink: int = 32,
        speech_window: int = 512,
        slide_delay: int = 32,
        slide_rate: int = 8,
        output_sink: int = 8,
        output_window: int = 512,
        output_evict_block: int = 1,
        speech_evict_block: int = 1,
    ):
        super().__init__(layers=[])
        self.speech_sink = speech_sink
        self.speech_window = speech_window
        self.slide_delay = slide_delay
        self.slide_rate = slide_rate
        self.output_sink = output_sink
        self.output_window = output_window
        self.output_evict_block = max(1, output_evict_block)
        self.speech_evict_block = max(1, speech_evict_block)
        self._speech_region: Optional[tuple[Any, Any]] = None

    def set_speech_region(self, start, end) -> None:
        self._speech_region = (start, end)
        for layer in self.layers:
            layer.set_speech_region(start, end)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        while len(self.layers) <= layer_idx:
            layer = SpeechStreamLayer(
                self.speech_sink, self.speech_window, self.slide_delay,
                self.slide_rate, self.output_sink, self.output_window,
                self.output_evict_block, self.speech_evict_block,
            )
            if self._speech_region is not None:
                layer.set_speech_region(*self._speech_region)
            self.layers.append(layer)
        return self.layers[layer_idx].update(key_states, value_states, cache_kwargs)

    @property
    def eviction_count(self) -> int:
        return sum(getattr(l, "eviction_count", 0) for l in self.layers)

    @property
    def compression_stats(self) -> tuple[int, int]:
        """(physical_kv, logical_kv) summed across layers. See StreamingCache."""
        physical = sum(
            (l.keys.shape[-2] if getattr(l, "is_initialized", False) else 0)
            for l in self.layers
        )
        logical = sum(getattr(l, "cumulative_length", 0) for l in self.layers)
        return physical, logical


def make_cache(policy: str | None, **kwargs) -> Cache | None:
    """Returns ``None`` for no policy (HF falls back to ``DynamicCache``)."""
    if policy is None:
        return None
    if policy == "streaming":
        return StreamingCache(
            sink_size=int(kwargs.get("sink_size", 4)),
            window_size=int(kwargs.get("window_size", 1024)),
            evict_block=int(kwargs.get("output_evict_block", 1)),
        )
    if policy == "speech_window":
        return SpeechWindowCache(
            speech_sink=int(kwargs.get("speech_sink", 32)),
            speech_window=int(kwargs.get("speech_window", 512)),
            slide_delay=int(kwargs.get("slide_delay", 32)),
            slide_rate=int(kwargs.get("slide_rate", 8)),
            speech_evict_block=int(kwargs.get("speech_evict_block", 1)),
            layer_ranges=kwargs.get("layer_ranges"),
        )
    if policy == "speech_stream":
        return SpeechStreamCache(
            speech_sink=int(kwargs.get("speech_sink", 32)),
            speech_window=int(kwargs.get("speech_window", 512)),
            slide_delay=int(kwargs.get("slide_delay", 32)),
            slide_rate=int(kwargs.get("slide_rate", 8)),
            output_sink=int(kwargs.get("output_sink", 8)),
            output_window=int(kwargs.get("output_window", 512)),
            output_evict_block=int(kwargs.get("output_evict_block", 1)),
            speech_evict_block=int(kwargs.get("speech_evict_block", 1)),
        )
    raise ValueError(f"Unknown cache policy: {policy!r}")
