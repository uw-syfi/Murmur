from __future__ import annotations

import io
import time

import numpy as np
import soundfile as sf
import torch

from dataclasses import dataclass, field

VIBEVOICE_SR = 24_000


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class TranscriptSegment:
    speaker_id: str
    start_s: float
    end_s: float
    text: str

    def as_dict(self) -> dict:
        return {
            "Speaker ID": self.speaker_id,
            "Start time": self.start_s,
            "End time": self.end_s,
            "Content": self.text,
        }


@dataclass
class InferenceStats:
    num_chunks: int = 0
    num_errors: int = 0
    inference_time_s: float = 0.0
    total_time_s: float = 0.0
    rtf: float = 0.0  # inference_time_s / audio_duration_s
    stage_times: dict = field(default_factory=dict)  # per-stage breakdown (s)

    def __str__(self) -> str:
        base = (
            f"chunks={self.num_chunks} errors={self.num_errors} "
            f"inf={self.inference_time_s:.2f}s total={self.total_time_s:.2f}s "
            f"RTF={self.rtf:.3f}"
        )
        if self.stage_times:
            stages = " ".join(f"{k}={v:.2f}s" for k, v in self.stage_times.items())
            return f"{base} | {stages}"
        return base


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dominant_speaker(
    start: float,
    end: float,
    timeline: list[tuple[float, float, str]],
) -> str:
    """Speaker with the most temporal overlap in [start, end]."""
    best_spk, best_overlap = "unknown", 0.0
    for t0, t1, spk in timeline:
        overlap = min(end, t1) - max(start, t0)
        if overlap > best_overlap:
            best_overlap, best_spk = overlap, spk
    return best_spk if best_overlap > 0.0 else "unknown"


def _merge_consecutive(
    segments: list[TranscriptSegment],
    max_gap_s: float = 1.5,
) -> list[TranscriptSegment]:
    """Merge adjacent segments from the same speaker with gap ≤ max_gap_s."""
    if not segments:
        return []
    merged = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        if seg.speaker_id == prev.speaker_id and seg.start_s - prev.end_s <= max_gap_s:
            merged[-1] = TranscriptSegment(
                speaker_id=prev.speaker_id,
                start_s=prev.start_s,
                end_s=seg.end_s,
                text=f"{prev.text} {seg.text}".strip(),
            )
        else:
            merged.append(seg)
    return merged


def _load_audio(wav_bytes: bytes) -> np.ndarray:
    """WAV bytes (16 kHz from VAD chunker) → float32 at 24 kHz."""
    arr, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != VIBEVOICE_SR:
        import torchaudio
        arr = torchaudio.functional.resample(
            torch.from_numpy(arr).unsqueeze(0), sr, VIBEVOICE_SR
        ).squeeze(0).numpy()
    return arr


def _build_speaker_mapping(
    parsed: list[tuple[float, float, str]],
    diarization_timeline: list[tuple[float, float, str]],
    window_start: float | None = None,
    window_end: float | None = None,
) -> dict[str, str]:
    """For each local speaker_id in *parsed*, pick the diarization speaker
    with the largest total temporal overlap across all of that local id's
    segments. Returns {local_id: global_id}. Optionally restrict the
    timeline to [window_start, window_end] for efficiency.
    """
    if window_start is not None and window_end is not None:
        tl = [
            (t0, t1, gspk) for t0, t1, gspk in diarization_timeline
            if t1 > window_start and t0 < window_end
        ]
    else:
        tl = diarization_timeline
    per_local: dict[str, dict[str, float]] = {}
    for abs_s, abs_e, local_spk in parsed:
        bucket = per_local.setdefault(local_spk, {})
        for t0, t1, gspk in tl:
            ov = min(abs_e, t1) - max(abs_s, t0)
            if ov > 0:
                bucket[gspk] = bucket.get(gspk, 0.0) + ov
    return {
        local_spk: (max(overlaps, key=overlaps.get) if overlaps else "unknown")
        for local_spk, overlaps in per_local.items()
    }


def _parse_ts(ts) -> float:
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, str):
        parts = ts.split(":")
        try:
            if len(parts) == 3:
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            return float(ts)
        except ValueError:
            return 0.0
    return 0.0


class LocalBatchEngine:
    """Batch VibeVoice ASR inference over VAD chunks."""

    def __init__(
        self,
        model,
        processor,
        device: str = "cuda",
        batch_size: int = 8,
        max_new_tokens: int = 32_768,
        merge_gap_s: float = 1.5,
        cache_policy: str | None = None,
        cache_kwargs: dict | None = None,
    ):
        self.model = model
        self.processor = processor
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.merge_gap_s = merge_gap_s
        self.cache_policy = cache_policy
        self.cache_kwargs = cache_kwargs or {}
        self._speech_engine_logged = False  # log decode engine once
        self.attn_implementation: str | None = None  # set by from_pretrained
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: str = "cuda",
        batch_size: int = 8,
        max_new_tokens: int = 32_768,
        attn_implementation: str = "flash_attention_2",
        dtype: torch.dtype = torch.bfloat16,
        merge_gap_s: float = 1.5,
        cache_policy: str | None = None,
        cache_kwargs: dict | None = None,
        **_,
    ) -> "LocalBatchEngine":
        from murmur.modeling.vibevoice import (
            VibeVoiceASRForConditionalGeneration,
            VibeVoiceASRProcessor,
        )

        processor = VibeVoiceASRProcessor.from_pretrained(
            model_path,
            language_model_pretrained_name="Qwen/Qwen2.5-7B",
        )
        model = VibeVoiceASRForConditionalGeneration.from_pretrained(
            model_path,
            dtype=dtype,
            attn_implementation=attn_implementation,
            trust_remote_code=True,
        )
        engine = cls(
            model=model, processor=processor, device=device,
            batch_size=batch_size, max_new_tokens=max_new_tokens,
            merge_gap_s=merge_gap_s,
            cache_policy=cache_policy, cache_kwargs=cache_kwargs,
        )
        engine.attn_implementation = attn_implementation
        return engine

    # ── File-based path ───────────────────────────────────────────────────────

    def transcribe(
        self,
        audio_path: str,
        diarization_timeline: list[tuple[float, float, str]],
        max_chunk_s: float = 30.0,
    ) -> tuple[list[TranscriptSegment], InferenceStats]:
        from murmur.chunking.vad_chunker import get_chunks_from_timeline

        t_total = time.perf_counter()
        chunks = get_chunks_from_timeline(audio_path, diarization_timeline, max_chunk_s)
        if not chunks:
            return [], InferenceStats()

        audio_duration_s = chunks[-1].end_s
        arrays = [_load_audio(c.wav_bytes) for c in chunks]

        t_inf = time.perf_counter()
        all_raw: list[list[dict]] = []
        for i in range(0, len(chunks), self.batch_size):
            raws, _ = self._generate(arrays[i : i + self.batch_size])
            all_raw.extend(raws)
        inf_elapsed = time.perf_counter() - t_inf

        segs: list[TranscriptSegment] = []
        for chunk, raw in zip(chunks, all_raw):
            for s in raw:
                rel_start = _parse_ts(s.get("start_time", s.get("Start time", 0.0)))
                rel_end   = _parse_ts(s.get("end_time",   s.get("End time",   chunk.end_s - chunk.start_s)))
                abs_start = max(chunk.start_s, min(chunk.start_s + rel_start, chunk.end_s))
                abs_end   = max(chunk.start_s, min(chunk.start_s + rel_end,   chunk.end_s))
                spk = _dominant_speaker(abs_start, abs_end, diarization_timeline)
                text = str(s.get("text", s.get("Content", ""))).strip()
                if text:
                    segs.append(TranscriptSegment(spk, abs_start, abs_end, text))

        segs.sort(key=lambda s: s.start_s)
        segs = _merge_consecutive(segs, self.merge_gap_s)

        total_elapsed = time.perf_counter() - t_total
        return segs, InferenceStats(
            num_chunks=len(chunks),
            inference_time_s=inf_elapsed,
            total_time_s=total_elapsed,
            rtf=inf_elapsed / audio_duration_s if audio_duration_s > 0 else 0.0,
        )

    # ── In-memory tensor path ─────────────────────────────────────────────────

    def transcribe_tensor(
        self,
        wav: torch.Tensor,
        max_chunk_s: float = 30.0,
        min_silence_ms: int = 500,
        min_speech_ms: int = 250,
        diarization_timeline: list[tuple[float, float, str]] | None = None,
    ) -> tuple[list[TranscriptSegment], InferenceStats]:
        from murmur.chunking.vad_chunker import iter_chunks_tensor

        t_total = time.perf_counter()

        t_vad = time.perf_counter()
        chunks = list(iter_chunks_tensor(
            wav, max_chunk_s, min_silence_ms, min_speech_ms,
            diarization_timeline=diarization_timeline,
            target_sr=VIBEVOICE_SR,
        ))
        vad_elapsed = time.perf_counter() - t_vad

        audio_duration_s = max((e for _, e, _ in chunks), default=0.0)

        t_inf = time.perf_counter()
        all_results: list[tuple[float, float, list[dict]]] = []
        preprocess_s = 0.0
        generate_s = 0.0
        decode_s = 0.0
        tokens_generated = 0.0
        max_tokens_budget_sum = 0.0
        evictions_total = 0.0
        cache_budget = 0.0
        cache_physical_total = 0.0
        cache_logical_total = 0.0
        num_batches = 0
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            arrays = [arr for _, _, arr in batch]
            raws, timings = self._generate(arrays)
            preprocess_s += timings["preprocess"]
            generate_s += timings["generate"]
            decode_s += timings["decode"]
            tokens_generated += timings.get("tokens_generated", 0.0)
            max_tokens_budget_sum += timings.get("max_tokens_budget", 0.0)
            evictions_total += timings.get("evictions", 0.0)
            cache_physical_total += timings.get("cache_physical_kv", 0.0)
            cache_logical_total += timings.get("cache_logical_kv", 0.0)
            # Cache budget is constant across batches for a given policy.
            cache_budget = timings.get("cache_budget", cache_budget)
            num_batches += 1
            for (s, e, _), raw in zip(batch, raws):
                all_results.append((s, e, raw))
        inf_elapsed = time.perf_counter() - t_inf

        t_align = time.perf_counter()
        segs: list[TranscriptSegment] = []
        for start_s, end_s, raw in all_results:
            # Parse this chunk's segments once.
            parsed: list[tuple[float, float, str, str]] = []
            for s in raw:
                rel_start = _parse_ts(s.get("start_time", s.get("Start time", 0.0)))
                rel_end   = _parse_ts(s.get("end_time",   s.get("End time",   end_s - start_s)))
                abs_start = max(start_s, min(start_s + rel_start, end_s))
                abs_end   = max(start_s, min(start_s + rel_end,   end_s))
                local_spk = str(s.get("speaker_id", s.get("Speaker ID", "unknown")))
                text = str(s.get("text", s.get("Content", ""))).strip()
                parsed.append((abs_start, abs_end, local_spk, text))

            # Per-chunk speaker remap: VibeVoice's local IDs ("0", "1", ...) are
            # consistent within a chunk but arbitrary across chunks.
            if diarization_timeline:
                mapping = _build_speaker_mapping(
                    [(a, b, lid) for a, b, lid, _ in parsed],
                    diarization_timeline,
                    window_start=start_s, window_end=end_s,
                )
            else:
                mapping = None

            for abs_start, abs_end, local_spk, text in parsed:
                if not text:
                    continue
                spk = mapping.get(local_spk, "unknown") if mapping is not None else local_spk
                segs.append(TranscriptSegment(spk, abs_start, abs_end, text))

        segs.sort(key=lambda s: s.start_s)
        segs = _merge_consecutive(segs, self.merge_gap_s)
        align_elapsed = time.perf_counter() - t_align

        total_elapsed = time.perf_counter() - t_total
        return segs, InferenceStats(
            num_chunks=len(all_results),
            inference_time_s=inf_elapsed,
            total_time_s=total_elapsed,
            rtf=inf_elapsed / audio_duration_s if audio_duration_s > 0 else 0.0,
            stage_times={
                "vad_chunking": vad_elapsed,
                "batch_preprocess": preprocess_s,
                "model_generate": generate_s,
                "decode_postprocess": decode_s,
                "align_merge": align_elapsed,
                "num_batches": float(num_batches),
                "tokens_generated": tokens_generated,
                "max_tokens_budget_sum": max_tokens_budget_sum,
                "cache_evictions": evictions_total,
                "cache_budget": cache_budget,
                "cache_physical_kv": cache_physical_total,
                "cache_logical_kv": cache_logical_total,
            },
        )

    # ── Direct transcription (no VAD / no chunking) ───────────────────────────

    def transcribe_audio(
        self,
        audio: "torch.Tensor | np.ndarray",
        src_sr: int = 16_000,
        diarization_timeline: list[tuple[float, float, str]] | None = None,
    ) -> tuple[list[TranscriptSegment], InferenceStats]:
        """Transcribe raw audio in a single forward pass — no VAD, no chunking.

        When ``diarization_timeline`` is provided, VibeVoice's emitted local
        speaker IDs are remapped to diarization speakers using the same
        per-(local-id) → global-id majority-overlap rule as the chunked path.
        """
        t_total = time.perf_counter()

        if isinstance(audio, torch.Tensor):
            arr = audio.float().cpu().numpy()
        else:
            arr = np.asarray(audio, dtype=np.float32)

        if arr.ndim > 1:
            arr = arr.mean(axis=0)

        if src_sr != VIBEVOICE_SR:
            import torchaudio
            arr = torchaudio.functional.resample(
                torch.from_numpy(arr).unsqueeze(0), src_sr, VIBEVOICE_SR
            ).squeeze(0).numpy()

        audio_duration_s = len(arr) / VIBEVOICE_SR

        t_inf = time.perf_counter()
        raws, gen_timings = self._generate([arr])
        raw = raws[0]
        inf_elapsed = time.perf_counter() - t_inf

        # Parse once so we can build the speaker mapping before emitting.
        parsed: list[tuple[float, float, str, str]] = []
        for s in raw:
            start_s = _parse_ts(s.get("start_time", s.get("Start time", 0.0)))
            end_s   = _parse_ts(s.get("end_time",   s.get("End time",   audio_duration_s)))
            local_spk = str(s.get("speaker_id", s.get("Speaker ID", "unknown")))
            text    = str(s.get("text", s.get("Content", ""))).strip()
            parsed.append((start_s, end_s, local_spk, text))

        if diarization_timeline:
            mapping = _build_speaker_mapping(
                [(a, b, lid) for a, b, lid, _ in parsed],
                diarization_timeline,
            )
        else:
            mapping = None

        segs: list[TranscriptSegment] = []
        for start_s, end_s, local_spk, text in parsed:
            if not text:
                continue
            spk = mapping.get(local_spk, "unknown") if mapping is not None else local_spk
            segs.append(TranscriptSegment(spk, start_s, end_s, text))

        total_elapsed = time.perf_counter() - t_total
        return segs, InferenceStats(
            num_chunks=1,
            inference_time_s=inf_elapsed,
            total_time_s=total_elapsed,
            rtf=inf_elapsed / audio_duration_s if audio_duration_s > 0 else 0.0,
            stage_times={
                "batch_preprocess": gen_timings.get("preprocess", 0.0),
                "model_generate": gen_timings.get("generate", 0.0),
                "decode_postprocess": gen_timings.get("decode", 0.0),
                "tokens_generated": gen_timings.get("tokens_generated", 0.0),
                "max_tokens_budget_sum": gen_timings.get("max_tokens_budget", 0.0),
                "cache_evictions": gen_timings.get("evictions", 0.0),
                "cache_budget": gen_timings.get("cache_budget", 0.0),
                "cache_physical_kv": gen_timings.get("cache_physical_kv", 0.0),
                "cache_logical_kv": gen_timings.get("cache_logical_kv", 0.0),
            },
        )

    # ── Core generation ───────────────────────────────────────────────────────

    def _generate(
        self, arrays: list[np.ndarray]
    ) -> tuple[list[list[dict]], dict[str, float]]:
        """Encode a batch of audio arrays and run model.generate.

        Returns (results, timings) where timings keys are
        'preprocess', 'generate', 'decode'.
        """
        timings = {
            "preprocess": 0.0, "generate": 0.0, "decode": 0.0,
            "tokens_generated": 0.0, "max_tokens_budget": 0.0,
            "evictions": 0.0,
        }

        longest_s = max(len(a) for a in arrays) / VIBEVOICE_SR
        # Decode runs at ~9 tokens/s of audio; budgeting 12 tok/s leaves ~33%
        # headroom and caps a derailed chunk near the real ceiling.
        max_tokens = min(self.max_new_tokens, max(256, int(longest_s * 12)))

        t = time.perf_counter()
        inputs = self.processor(
            audio=arrays,
            sampling_rate=VIBEVOICE_SR,
            return_tensors="pt",
            padding=True,
            add_generation_prompt=True,
        )
        inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings["preprocess"] = time.perf_counter() - t

        # Fresh cache per batch — they're stateful and tied to one batch shape.
        from murmur.inference.kv_cache import (
            make_cache, SpeechWindowCache, SpeechStreamCache,
        )
        past_kv = make_cache(self.cache_policy, **self.cache_kwargs)
        if isinstance(past_kv, (SpeechWindowCache, SpeechStreamCache)):
            # Locate each row's speech span. Left-padding shifts every row's
            # <|speech_start|>..<|speech_end|> to a different offset, so we
            # record the boundaries per row.
            sid = self.processor.speech_start_id
            eid = self.processor.speech_end_id
            starts: list[int] = []
            ends: list[int] = []
            for b in range(inputs["input_ids"].shape[0]):
                row = inputs["input_ids"][b].tolist()
                try:
                    sstart = row.index(sid) + 1  # first speech-pad after marker
                    send = row.index(eid, sstart)  # position of <|speech_end|>
                except ValueError as e:
                    raise ValueError(
                        f"{self.cache_policy}: could not locate "
                        f"<|speech_start|>/<|speech_end|> tokens in row {b}"
                    ) from e
                starts.append(sstart)
                ends.append(send)
            past_kv.set_speech_region(starts, ends)
            timings["speech_span"] = float(
                min(e - s for s, e in zip(starts, ends))
            )
            if not self._speech_engine_logged:
                print(
                    f"[{self.cache_policy}] decoding on model.generate. "
                    "Batched speech eviction runs here directly."
                )
                self._speech_engine_logged = True

        t = time.perf_counter()
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                past_key_values=past_kv,
                max_new_tokens=max_tokens,
                pad_token_id=self.processor.pad_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                do_sample=False,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings["generate"] = time.perf_counter() - t
        timings["max_tokens_budget"] = float(max_tokens)
        timings["tokens_generated"] = float(output_ids.shape[1] - inputs["input_ids"].shape[1])
        if past_kv is not None and hasattr(past_kv, "eviction_count"):
            timings["evictions"] = float(past_kv.eviction_count)
        if past_kv is not None and hasattr(past_kv, "compression_stats"):
            phys, logi = past_kv.compression_stats
            timings["cache_physical_kv"] = float(phys)
            timings["cache_logical_kv"] = float(logi)
        # Cache budget (sink + window) — 0 means no policy / not bounded.
        if self.cache_policy == "streaming":
            timings["cache_budget"] = float(
                self.cache_kwargs.get("sink_size", 4)
                + self.cache_kwargs.get("window_size", 1024)
            )
        elif self.cache_policy == "speech_window":
            # With per-layer overrides the budget varies by layer, so a single
            # number is misleading — leave it at 0 to suppress the print.
            if not self.cache_kwargs.get("layer_ranges"):
                timings["cache_budget"] = float(
                    self.cache_kwargs.get("speech_sink", 32)
                    + self.cache_kwargs.get("speech_window", 512)
                )
        elif self.cache_policy == "speech_stream":
            timings["cache_budget"] = float(
                self.cache_kwargs.get("speech_sink", 32)
                + self.cache_kwargs.get("speech_window", 512)
                + self.cache_kwargs.get("output_sink", 8)
                + self.cache_kwargs.get("output_window", 512)
            )

        t = time.perf_counter()
        input_len = inputs["input_ids"].shape[1]
        results: list[list[dict]] = []
        for i in range(output_ids.shape[0]):
            gen = output_ids[i, input_len:]
            eos = (gen == self.processor.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos):
                gen = gen[: eos[0] + 1]
            text = self.processor.decode(gen, skip_special_tokens=True)
            try:
                segs = self.processor.post_process_transcription(text)
            except Exception:
                segs = [{"text": text, "start_time": 0.0, "end_time": 0.0, "speaker_id": "unknown"}]
            results.append(segs)
        timings["decode"] = time.perf_counter() - t

        return results, timings
