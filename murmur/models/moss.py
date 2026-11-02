"""MOSS-Transcribe-Diarize (OpenMOSS-Team/MOSS-Transcribe-Diarize) on a vLLM nightly build.

The model writes ``[start][Sxx]text[end]`` segments, e.g. ``[0.00][S01]hello there[1.84]``.
"""

from __future__ import annotations

import io
import re

import numpy as np
import requests

from .base import LocalSegment, ModelClient, Reply

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


class MossClient(ModelClient):
    name = "moss"
    hf_id = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
    sample_rate = 16_000
    draft_windows_default = False  # 0.6B decoder: drafting costs more than it saves (paper Sec. 4.3)
    merge_gap_s = None
    tokens_per_s = 32.0
    min_tokens = 512

    def budget(self, audio):
        return max(self.min_tokens, int(len(audio) / self.sample_rate * self.tokens_per_s))

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

    def parse(self, text, keep_raw: bool = False):
        return [LocalSegment(s, e, spk, t) for s, e, spk, t in parse_moss_transcript(text)]

    def flagged(self, reply: Reply) -> bool:
        if reply.failed:
            return True
        text = reply.text
        cut_off = bool(text.strip()) and not _CLOSED_RE.search(text)
        return cut_off or (bool(text.strip()) and not parse_moss_transcript(text))

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
