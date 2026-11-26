"""MOSS-Transcribe-Diarize (OpenMOSS-Team/MOSS-Transcribe-Diarize) on vLLM and Apple MLX.

The model writes ``[start][Sxx]text[end]`` segments, e.g. ``[0.00][S01]hello there[1.84]``.
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from threading import Lock
from time import perf_counter

import numpy as np
import requests

from .base import DiarizingModel, LocalSegment, Reply

DEFAULT_CHECKPOINT = "vanch007/mlx-MOSS-Transcribe-Diarize"

PACKED_CHECKPOINTS = {
    "int4": "vanch007/mlx-MOSS-Transcribe-Diarize-4bit",
    "int8": "vanch007/mlx-MOSS-Transcribe-Diarize-8bit",
}

# (MLX mode, bits, group size) for weight-only quantization at load time.
MLX_QUANTIZATION = {
    **{f"int{bits}": ("affine", bits, 64) for bits in (1, 2, 3, 4, 5, 6, 8)},
    "mxfp4": ("mxfp4", 4, 32),
    "mxfp8": ("mxfp8", 8, 32),
    "nvfp4": ("nvfp4", 4, 16),
}

_SEG_RE = re.compile(r"\[(\d+(?:\.\d+)?)\]\[(S\d+)\](.*?)\[(\d+(?:\.\d+)?)\]"
                     r"(?=\s*\[\d+(?:\.\d+)?\]\[S\d+\]|\s*$)", re.DOTALL)
_CLOSED_RE = re.compile(r"\[\d+(?:\.\d+)?\]\s*$")


def parse_moss_transcript(text: str) -> list[tuple[float, float, str, str]]:
    """``(start, end, speaker, text)`` tuples from MOSS's raw output."""
    try:
        from moss_transcribe_diarize import parse_transcript  # reference parser, if installed
        return [(float(s.start), float(s.end), s.speaker, s.text) for s in parse_transcript(text)]
    except ImportError:
        out = []
        for m in _SEG_RE.finditer(text):
            seg_text = m.group(3).strip()
            if seg_text:
                out.append((float(m.group(1)), float(m.group(4)), m.group(2), seg_text))
        return out


class MossModel(DiarizingModel):
    name = "moss"
    hf_id = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
    sample_rate = 16_000
    draft_windows_default = False  # 0.6B decoder: drafting costs more than it saves (paper Sec. 4.3)
    merge_gap_s = None
    drafts_use_diarization = True  # drafts the next speaker tag from the diarization
    max_audio_s = 5400.0  # 90 min
    tokens_per_s = 32.0
    min_tokens = 512

    def budget(self, audio):
        return max(self.min_tokens, int(len(audio) / self.sample_rate * self.tokens_per_s))

    def parse(self, text, keep_raw: bool = False):
        return [LocalSegment(s, e, spk, t) for s, e, spk, t in parse_moss_transcript(text)]

    def flagged(self, reply: Reply) -> bool:
        if reply.failed:
            return True
        text = reply.text
        cut_off = bool(text.strip()) and not _CLOSED_RE.search(text)
        return cut_off or (bool(text.strip()) and not parse_moss_transcript(text))


class MossClient(MossModel):
    def _post(self, audio, max_tokens, headers):
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, np.asarray(audio, dtype=np.float32), self.sample_rate, format="WAV", subtype="PCM_16")
        r = requests.post(
            f"{self.url}/v1/audio/transcriptions",
            files={"file": ("chunk.wav", buf.getvalue(), "audio/wav")},
            data={"model": self.served_name, "response_format": "json", "temperature": "0",
                  "max_completion_tokens": str(max_tokens), "repetition_penalty": "1.0"},
            headers=headers, timeout=self.timeout)
        r.raise_for_status()
        return r.json().get("text", "") or "", None

    @classmethod
    def server_args(cls, batch_size, speculative_config, single_pass: bool = False, gpu_memory_utilization=None):
        args = ["--trust-remote-code", "--max-num-seqs", str(batch_size),
                "--max-num-batched-tokens", "131072" if single_pass else "49152",
                "--gpu-memory-utilization", str(gpu_memory_utilization or 0.70)]
        if single_pass:
            args.append("--enable-chunked-prefill")
        if speculative_config:
            args += ["--speculative-config", speculative_config]
        env = {"VLLM_MAX_AUDIO_CLIP_FILESIZE_MB": "300", "VLLM_MAX_AUDIO_DECODE_DURATION_S": "4000"}
        return args, env


class MLXMossClient(MossModel):
    def __init__(self, checkpoint: str | Path | None = None, quantization: str | None = None):
        if quantization is not None and quantization not in MLX_QUANTIZATION:
            raise ValueError(f"Unknown MLX quantization {quantization!r}. Choose from {', '.join(MLX_QUANTIZATION)}.")
        try:
            from moss_transcribe_diarize.mlx import load_model
        except ImportError as exc:
            raise ImportError("Install the MOSS MLX runtime as described in the README.") from exc
        if checkpoint is None and quantization in PACKED_CHECKPOINTS:
            checkpoint = PACKED_CHECKPOINTS[quantization]
            quantization = None  # The checkpoint config describes the packed layers.
        self.model = load_model(checkpoint or DEFAULT_CHECKPOINT, strict=True)
        if quantization is not None:
            import mlx.core as mx
            import mlx.nn as nn

            if any(isinstance(layer, (nn.QuantizedLinear, nn.QuantizedEmbedding)) for _, layer in self.model.named_modules()):
                raise ValueError("Checkpoint is already quantized. Omit quantization to use its stored weights.")
            mode, bits, group_size = MLX_QUANTIZATION[quantization]
            nn.quantize(self.model, mode=mode, bits=bits, group_size=group_size,
                        class_predicate=self.model.model_quant_predicate)
            mx.eval(self.model.parameters())
        self._lock = Lock()

    def decode(self, audio, request_id=None, retries=3) -> Reply:
        with self._lock:
            if self.model is None:
                raise RuntimeError("The MLX model is closed.")
            start = perf_counter()
            result = self.model.generate(audio, max_tokens=self.budget(audio), temperature=0.0)
            return Reply(result.text, perf_counter() - start)

    def decode_with_draft(self, audio, words, language, diar_segments, chunk_offset) -> Reply:
        with self._lock:
            if self.model is None:
                raise RuntimeError("The MLX model is closed.")
            start = perf_counter()
            text, stats = _generate_with_draft(self.model, audio, words, language, diar_segments, chunk_offset, self.budget(audio))
            return Reply(text, perf_counter() - start, speculative=stats)

    def close(self) -> None:
        with self._lock:
            self.model = None


def verify_greedy(first, proposer, verify, rewind, eos, budget):
    """Keep the last emitted token pending. The cache holds everything before it."""
    tokens = []
    stats = {"proposed": 0, "accepted": 0, "verification_calls": 0}
    if budget <= 0 or first in eos:
        return tokens, stats
    tokens.append(first)
    while len(tokens) < budget:
        # Leave room for the target's correction (or a bonus after a full match).
        k = min(8, budget - len(tokens) - 1)
        draft = proposer.propose(tokens, k)[:k] if k else []
        predictions = verify([tokens[-1], *draft])
        matched = 0
        while matched < len(draft) and predictions[matched] == draft[matched]:
            matched += 1
        stats["verification_calls"] += 1
        stats["proposed"] += len(draft)
        stats["accepted"] += matched
        if matched < len(draft):
            rewind(len(draft) - matched)
        for token in predictions[:matched + 1]:
            if token in eos:
                return tokens, stats
            tokens.append(token)
    return tokens, stats


def _generate_with_draft(model, audio, words, language, diar_segments, offset, budget):
    import mlx.core as mx
    from mlx_lm.models.cache import trim_prompt_cache

    from ..drafting.vllm_proposer import DraftTranscript, MossDrafter

    tokenizer = model._tokenizer
    transcript = DraftTranscript(words, tokenizer, sep="" if language == "zh" else " ")
    proposer = MossDrafter(transcript, tokenizer, language, diar_segments, offset)
    ids, embeddings, length, _ = model._prepare_generation_inputs(audio, None)
    cache = model.make_cache()

    # Mirror the runtime's chunked audio prefill, leaving the final prompt token
    # for a one-position forward pass just like the ordinary greedy generator.
    for start in range(0, length - 1, 2048):
        end = min(start + 2048, length - 1)
        model(ids[None, start:end], input_embeddings=embeddings[None, start:end], cache=cache)
        mx.eval([layer.state for layer in cache])
    logits = model(ids[None, -1:], input_embeddings=embeddings[None, -1:], cache=cache)
    first = mx.argmax(logits[0, -1]).item()

    def verify(block):
        logits = model(mx.array([block]), cache=cache)
        return mx.argmax(logits[0], axis=-1).tolist()

    def rewind(count):
        if trim_prompt_cache(cache, count) != count:
            raise RuntimeError("MOSS could not rewind the speculative token cache.")

    tokens, stats = verify_greedy(first, proposer, verify, rewind, model._eos_token_ids(), budget)
    return tokenizer.decode(tokens, skip_special_tokens=True).strip(), stats
