"""Murmur ASR models served by llama.cpp's multimodal GGUF runtime."""

from __future__ import annotations

import base64
import math
from functools import cached_property

import requests

from .vibevoice import SHOW_KEYS, VibeVoiceClient, _wav_bytes
from .voxtral import VoxtralClient, VoxtralSmallClient


class LlamaContext:
    """Size each llama-server slot for the longest audio one request carries.

    A slot holds the prompt, the audio embeddings and the output budget. Sizing it for a
    target chunk instead of the model's longest single pass saves gigabytes of KV cache on
    laptops. ``context_s`` is the audio length the server was sized for (None for a
    server Murmur did not start, whose size it cannot know).
    """
    audio_tokens_per_s: float
    prompt_tokens = 512
    context_s: float | None = None

    @classmethod
    def output_tokens(cls, audio_s: float) -> int:
        return cls.budget_for(audio_s)

    @classmethod
    def context_for(cls, audio_s: float) -> int:
        need = cls.prompt_tokens + math.ceil(audio_s * cls.audio_tokens_per_s) + cls.output_tokens(audio_s)
        return 1024 * math.ceil(need / 1024)

    def _check_length(self, audio):
        duration = len(audio) / self.sample_rate
        if self.context_s is not None and duration > self.context_s + 1.0:
            raise ValueError(f"{duration:.0f} s of audio does not fit a llama.cpp server sized for {self.context_s:.0f} s "
                             "chunks; start it with single_pass_server=True (CLI: --single-pass) for single passes")


class LlamaVibeVoiceClient(LlamaContext, VibeVoiceClient):
    gguf_repo = "ggml-org/VibeVoice-ASR-GGUF"
    audio_tokens_per_s = 7.5  # measured about 6.9 per second of audio, plus margin
    # Normal output is about 5 tokens per second of audio. vLLM allows 100 per second (32k for
    # a 330 s chunk) to let repetition loops run out; at laptop speed such a loop would decode
    # for half an hour, so llama.cpp stops one at 25 per second (about 8k tokens per chunk).
    output_tokens_per_s = 25

    @classmethod
    def budget_for(cls, audio_s):
        return min(cls.max_new_tokens, max(256, int(audio_s * cls.output_tokens_per_s)))

    def budget(self, audio):
        return self.budget_for(len(audio) / self.sample_rate)

    def _post(self, audio, max_tokens, headers):
        self._check_length(audio)
        encoded = base64.b64encode(_wav_bytes(audio, self.sample_rate)).decode("ascii")
        duration = len(audio) / self.sample_rate
        prompt = f"This is a {duration:.2f} seconds audio, please transcribe it with these keys: " + ", ".join(SHOW_KEYS)
        content = [
            {"type": "input_audio", "input_audio": {"data": encoded, "format": "wav"}},
            {"type": "text", "text": prompt},
        ]
        response = requests.post(
            f"{self.url}/v1/chat/completions",
            json={"model": self.served_name, "messages": [{"role": "user", "content": content}],
                  "temperature": 0, "max_tokens": max_tokens, "stream": False},
            headers=headers, timeout=self.timeout,
        )
        response.raise_for_status()
        choice = response.json()["choices"][0]
        return choice["message"].get("content") or "", choice.get("finish_reason")


class LlamaVoxtralClient(LlamaContext, VoxtralClient):
    gguf_repo = "ggml-org/Voxtral-Mini-3B-2507-GGUF"
    audio_tokens_per_s = 7.5  # measured about 5.6 per second of audio, plus margin

    @classmethod
    def budget_for(cls, audio_s):
        return int(audio_s * cls.tokens_per_s) + cls.min_tokens

    @cached_property
    def media_marker(self):
        # llama-server generates a marker at startup. Its value is specific to the server instance.
        response = requests.get(f"{self.url}/props", params={"model": self.served_name}, timeout=self.timeout)
        response.raise_for_status()
        marker = response.json().get("media_marker")
        if not marker:
            raise ValueError("llama-server must expose an audio projector and media_marker in /props")
        return marker

    def _post(self, audio, max_tokens, headers):
        self._check_length(audio)
        # Mistral's transcription mode places the language and [TRANSCRIBE] after [/INST].
        # libmtmd supplies [BEGIN_AUDIO] when replacing the marker with the audio embeddings.
        prompt = f"[INST]{self.media_marker}[/INST]lang:{self.language}[TRANSCRIBE]"
        encoded = base64.b64encode(_wav_bytes(audio, self.sample_rate)).decode("ascii")
        response = requests.post(
            f"{self.url}/completion",
            json={"model": self.served_name, "prompt": {"prompt_string": prompt, "multimodal_data": [encoded]},
                  "temperature": 0, "n_predict": max_tokens, "stream": False},
            headers=headers, timeout=self.timeout,
        )
        response.raise_for_status()
        result = response.json()
        finish = "length" if result.get("stop_type") == "limit" or result.get("truncated") else "stop"
        return result["content"], finish


class LlamaVoxtralSmallClient(LlamaVoxtralClient):
    name = VoxtralSmallClient.name
    hf_id = VoxtralSmallClient.hf_id
    gguf_repo = "bartowski/mistralai_Voxtral-Small-24B-2507-GGUF"


def llama_client(model):
    for client in (LlamaVibeVoiceClient, LlamaVoxtralClient, LlamaVoxtralSmallClient):
        if client.name == model.name:
            return client
    raise ValueError(f"llama.cpp supports vibevoice, voxtral and voxtral-small; got {model.name!r}")
