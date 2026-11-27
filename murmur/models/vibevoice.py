"""VibeVoice-ASR (microsoft/VibeVoice-ASR) on vLLM 0.14 with the VibeVoice vLLM plugin.

The model writes a JSON list of segments:
``[{"Start":0.0,"End":3.2,"Speaker":0,"Content":"..."}, ...]``.
"""

from __future__ import annotations

import base64
import io
import json
from time import perf_counter

import numpy as np
import requests

from .base import DiarizingModel, LocalSegment, Reply
from .mlx import MLXClient, verify_greedy

SYSTEM_PROMPT = "You are a helpful assistant that transcribes audio input into text output in JSON format."
SHOW_KEYS = ["Start time", "End time", "Speaker ID", "Content"]
_KEYS = {"Start time": "start", "Start": "start", "End time": "end", "End": "end",
         "Speaker ID": "speaker", "Speaker": "speaker", "Content": "text"}


def _wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, np.asarray(audio, dtype=np.float32), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def parse_time(ts) -> float:
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


def parse_json_transcript(text: str) -> list[dict] | None:
    """VibeVoice's JSON output as a list of dicts; None if it does not parse (e.g. a repetition loop)."""
    if "```json" in text:
        a = text.find("```json") + 7
        js = text[a:text.find("```", a)].strip()
    else:
        a = text.find("[")
        if a == -1:
            a = text.find("{")
        if a == -1:
            js = text
        else:
            depth, b = 0, a
            for i in range(a, len(text)):
                if text[i] in "[{":
                    depth += 1
                elif text[i] in "]}":
                    depth -= 1
                    if depth == 0:
                        b = i + 1
                        break
            js = text[a:b]
    try:
        out = json.loads(js)
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(out, dict):
        out = [out]
    rows = []
    for item in out if isinstance(out, list) else []:
        if isinstance(item, dict):
            row = {_KEYS[k]: v for k, v in item.items() if k in _KEYS}
            if row:
                rows.append(row)
    return rows


class VibeVoiceClient(DiarizingModel):
    name = "vibevoice"
    hf_id = "microsoft/VibeVoice-ASR"
    sample_rate = 24_000
    draft_windows_default = True
    merge_gap_s = 1.5
    clamp_to_chunk = True
    ngram_patch = "vibevoice"  # vLLM 0.14: no custom-proposer hook
    max_audio_s = 3600.0  # 60 min: longer single passes fail (empirical)
    max_new_tokens = 32_768

    def default_served_name(self) -> str:
        return "vibevoice"

    def budget(self, audio):
        return min(self.max_new_tokens, max(256, int(len(audio) / self.sample_rate * 100)))

    def _post(self, audio, max_tokens, headers):
        dur = len(audio) / self.sample_rate
        b64 = base64.b64encode(_wav_bytes(audio, self.sample_rate)).decode()
        payload = {
            "model": self.served_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{b64}"}},
                    {"type": "text", "text": f"This is a {dur:.2f} seconds audio, please transcribe it "
                                             f"with these keys: " + ", ".join(SHOW_KEYS)},
                ]},
            ],
            "max_tokens": max_tokens, "temperature": 0.0, "top_p": 1.0,
            "repetition_penalty": 1.0,  # raw greedy decoding
            "stream": False,
        }
        r = requests.post(f"{self.url}/v1/chat/completions", json=payload, headers=headers, timeout=self.timeout)
        r.raise_for_status()
        choice = r.json()["choices"][0]
        return choice["message"].get("content") or "", choice.get("finish_reason")

    def parse(self, text, keep_raw: bool = False):
        """Segments of one chunk. A missing end time means "until the chunk ends" (``end=None``).

        An output that does not parse (a repetition loop that never closes its JSON) gives no
        segments, so the chunk's words count as missed, as in the paper's evaluation.
        ``keep_raw`` returns it as one raw segment instead (for inspection).
        """
        rows = parse_json_transcript(text)
        if rows is None:
            return [LocalSegment(0.0, 0.0, "unknown", text)] if keep_raw and text.strip() else []
        return [LocalSegment(parse_time(r.get("start", 0.0)),
                             parse_time(r["end"]) if "end" in r else None,
                             str(r.get("speaker", "unknown")), str(r.get("text", "")).strip())
                for r in rows]

    def flagged(self, reply: Reply) -> bool:
        if reply.failed:
            return True
        rows = parse_json_transcript(reply.text)
        return reply.finish == "length" or rows is None or (not rows and bool(reply.text.strip()))

    @classmethod
    def model_path(cls, vllm_bin):
        """The local snapshot, with the tokenizer files the VibeVoice vLLM plugin needs.

        The model repo ships no tokenizer; VibeVoice's plugin generates one from Qwen2.5
        (vllm_plugin.tools.generate_tokenizer_files), run here in the server's environment.
        """
        import subprocess
        from pathlib import Path

        from huggingface_hub import snapshot_download
        path = snapshot_download(cls.hf_id)
        if not (Path(path) / "tokenizer.json").exists():
            server_python = str(Path(vllm_bin).resolve().parent / "python")
            subprocess.run([server_python, "-m", "vllm_plugin.tools.generate_tokenizer_files", "--output", path],
                           check=True)
        return path

    @classmethod
    def server_args(cls, batch_size, speculative_config, single_pass=False, gpu_memory_utilization=None):
        args = ["--served-model-name", "vibevoice", "--trust-remote-code", "--dtype", "bfloat16",
                "--max-num-seqs", str(batch_size),
                "--cudagraph-capture-sizes", *map(str, [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 144]),
                "--max-model-len", "65536", "--max-num-batched-tokens", "32768",
                # leave room for the client's pyannote and draft model on the same GPU
                "--gpu-memory-utilization", str(gpu_memory_utilization or 0.70),
                "--no-enable-prefix-caching", "--enable-chunked-prefill",
                "--chat-template-content-format", "openai"]
        if speculative_config:
            args += ["--speculative-config", speculative_config]
        return args, {}


class MLXVibeVoiceClient(MLXClient, VibeVoiceClient):
    def _load(self, checkpoint):
        from mlx_audio.stt.utils import load_model
        return load_model(checkpoint, strict=True)

    def decode_with_draft(self, audio, words, language, diar_segments, chunk_offset):
        from ..drafting.vllm_proposer import VibeVoiceDrafter

        with self._lock:
            if self.model is None:
                raise RuntimeError("The MLX model is closed.")
            if self.model.is_streaming_model:
                raise ValueError("VibeVoice draft verification requires an offline ASR checkpoint.")
            start = perf_counter()
            proposer = VibeVoiceDrafter.from_words(words, self.model.tokenizer, language)
            text, stats = _generate_with_draft(self.model, audio, proposer, self.budget(audio))
            return Reply(text, perf_counter() - start, speculative=stats)

    def decode_draft_batch(self, jobs, language):
        from ..drafting.vllm_proposer import VibeVoiceDrafter
        from .mlx import GreedySequence
        from .vibevoice_batch import verify_batch

        with self._lock:
            if self.model is None or self.model.is_streaming_model:
                raise ValueError("Batched verification requires an open offline VibeVoice model.")
            sequences, caches, starts = [], [], []
            eos = _stop_tokens(self.model)
            for audio, words in jobs:
                starts.append(perf_counter())
                first, cache = _prefill(self.model, audio)
                proposer = VibeVoiceDrafter.from_words(words, self.model.tokenizer, language)
                sequences.append(GreedySequence(first, proposer, eos, self.budget(audio)))
                caches.append(cache)
            finished = verify_batch(self.model.language_model, sequences, caches)
            return [Reply(self.model.tokenizer.decode(seq.tokens, skip_special_tokens=True).strip(),
                          end - start, speculative=seq.stats)
                    for seq, start, end in zip(sequences, starts, finished)]


def _prefill(model, audio):
    import mlx.core as mx
    from mlx_audio.lm.models.cache import make_prompt_cache

    # Match mlx-audio's speech prompt and 2048-token prefill before verifying draft blocks.
    waveform = model._preprocess_audio(audio, sampling_rate=24_000)
    speech = model.encode_speech(waveform)
    ids, mask = model._build_prompt_tokens(speech, waveform.shape[1] / 24_000)
    embeddings = model._merge_speech_text_embeddings(ids, speech, mask)
    lm = model.language_model
    cache = make_prompt_cache(lm)
    for start in range(0, ids.shape[1] - 1, 2048):
        end = min(start + 2048, ids.shape[1] - 1)
        lm(ids[:, start:end], cache=cache, input_embeddings=embeddings[:, start:end])
        mx.eval([layer.state for layer in cache])
        mx.clear_cache()
    logits = lm(ids[:, -1:], cache=cache, input_embeddings=embeddings[:, -1:])
    first = mx.argmax(logits[0, -1]).item()

    return first, cache


def _stop_tokens(model):
    return {151643, 151645, model.tokenizer.eos_token_id}  # endoftext and im_end


def _generate_with_draft(model, audio, proposer, budget):
    import mlx.core as mx

    first, cache = _prefill(model, audio)
    lm = model.language_model

    def verify(block):
        return mx.argmax(lm(mx.array([block]), cache=cache)[0], axis=-1).tolist()

    def rewind(count):
        for layer in cache:
            if layer.trim(count) != count:
                raise RuntimeError("VibeVoice could not rewind the speculative token cache.")

    eos = _stop_tokens(model)
    tokens, stats = verify_greedy(first, proposer, verify, rewind, eos, budget)
    mx.clear_cache()
    return model.tokenizer.decode(tokens, skip_special_tokens=True).strip(), stats
