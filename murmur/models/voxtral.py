"""Voxtral (mistralai/Voxtral-Mini-3B-2507, Voxtral-Small-24B-2507) on vLLM 0.14.

Voxtral writes plain text, so it is a :class:`TranscriptModel`: Murmur adds the word
times and speakers. Requests go to vLLM's transcription endpoint, which sends each
target chunk (or single-pass piece) to the model as one request: vLLM does not split
audio for Voxtral. Murmur's server patch (``murmur.drafting.vllm_proposer.allow_long_audio``,
enabled by MURMUR_LONG_AUDIO=1) is a safeguard that keeps it that way.
"""

from __future__ import annotations

import io

import numpy as np
import requests

from .base import TranscriptModel
from .mlx import MLXClient


class VoxtralClient(TranscriptModel):
    name = "voxtral"
    hf_id = "mistralai/Voxtral-Mini-3B-2507"
    sample_rate = 16_000
    # Murmur always runs the short-window ASR for a transcript model (word times),
    # so its drafts come at no extra cost.
    draft_windows_default = True
    max_audio_s = 1800.0  # 30 min: Voxtral's limit for transcription (32k-token context)
    tokens_per_s, min_tokens = 8, 256
    language = "en"  # set by Murmur.load(language=...)
    gpu_memory_utilization = 0.70  # leaves room for pyannote and the short-window ASR

    def budget(self, audio):
        return int(len(audio) / self.sample_rate * self.tokens_per_s) + self.min_tokens

    def _post(self, audio, max_tokens, headers):
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, np.asarray(audio, dtype=np.float32), self.sample_rate, format="WAV", subtype="PCM_16")
        r = requests.post(
            f"{self.url}/v1/audio/transcriptions",
            files={"file": ("chunk.wav", buf.getvalue(), "audio/wav")},
            data={"model": self.served_name, "language": self.language, "response_format": "json", "temperature": "0",
                  "max_completion_tokens": str(max_tokens)},
            headers=headers, timeout=self.timeout)
        r.raise_for_status()
        return r.json().get("text", "") or "", None

    @classmethod
    def server_args(cls, batch_size, speculative_config, single_pass=False, gpu_memory_utilization=None):
        args = ["--tokenizer-mode", "mistral", "--config-format", "mistral", "--load-format", "mistral",
                "--dtype", "bfloat16", "--max-model-len", "32768", "--max-num-seqs", str(batch_size),
                "--gpu-memory-utilization", str(gpu_memory_utilization or cls.gpu_memory_utilization), "--no-enable-prefix-caching"]
        if speculative_config:
            args += ["--speculative-config", speculative_config]
        return args, {"MURMUR_LONG_AUDIO": "1", "VLLM_MAX_AUDIO_CLIP_FILESIZE_MB": "512"}


class VoxtralSmallClient(VoxtralClient):
    name = "voxtral-small"
    hf_id = "mistralai/Voxtral-Small-24B-2507"
    gpu_memory_utilization = 0.85  # 48 GB of weights; pyannote and Whisper-tiny need ~3 GB


class MLXVoxtralClient(MLXClient, VoxtralClient):
    def _load(self, checkpoint):
        import mlx.core as mx
        from mlx_voxtral import VoxtralProcessor, load_voxtral_model

        model, _ = load_voxtral_model(checkpoint, dtype=mx.bfloat16)
        self.processor = VoxtralProcessor.from_pretrained(checkpoint)
        return model

    def _generate(self, audio, budget):
        inputs = self.processor.apply_transcrition_request(audio, language=self.language,
                                                          sampling_rate=self.sample_rate)
        # Float32 mel features would promote the encoder and the decoder KV cache to float32.
        features = inputs.input_features.astype(self.model.audio_tower.conv1.weight.dtype)
        ids = self.model.generate(input_ids=inputs.input_ids, input_features=features,
                                  max_new_tokens=budget, repetition_penalty=1.0, temperature=0.0)
        tokens = ids[0, inputs.input_ids.shape[1]:].tolist()
        text = self.processor.decode(tokens, skip_special_tokens=True)
        return text, "length" if len(tokens) == budget else None


class MLXVoxtralSmallClient(MLXVoxtralClient, VoxtralSmallClient):
    pass
