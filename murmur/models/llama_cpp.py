"""Murmur ASR models served by llama.cpp's multimodal GGUF runtime."""

from __future__ import annotations

import base64
from functools import cached_property

import requests

from .vibevoice import SHOW_KEYS, VibeVoiceClient, _wav_bytes
from .voxtral import VoxtralClient, VoxtralSmallClient


class LlamaVibeVoiceClient(VibeVoiceClient):
    gguf_repo = "ggml-org/VibeVoice-ASR-GGUF"
    context_per_slot = 65536

    def _post(self, audio, max_tokens, headers):
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


class LlamaVoxtralClient(VoxtralClient):
    gguf_repo = "ggml-org/Voxtral-Mini-3B-2507-GGUF"
    context_per_slot = 32768

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
