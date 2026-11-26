"""What Murmur needs to know about a long-context ASR model served by vLLM."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

import numpy as np
import requests


@dataclass
class LocalSegment:
    """A segment as the model wrote it: times relative to the chunk start, the model's own speaker label."""

    start: float
    end: float | None  # None: until the end of the chunk
    speaker: str
    text: str


@dataclass
class Reply:
    text: str
    seconds: float
    finish: str | None = None
    failed: bool = False  # HTTP request failed after retries
    speculative: dict[str, int] | None = None


class ModelClient:
    """Request format, output parsing and server settings for one model.

    Subclasses define: ``name``, ``hf_id``, ``sample_rate``, ``draft_windows_default``,
    ``budget()``, ``_post()``, ``parse()``, ``flagged()`` and ``server_args()``.
    Models derive from :class:`DiarizingModel` or :class:`TranscriptModel`.
    """

    #: The output has the model's own speaker labels and timestamps. If False, Murmur
    #: adds them (word times from the short-window ASR, speakers from diarization).
    diarizes: bool = True

    name: str = ""
    hf_id: str = ""
    sample_rate: int = 16_000
    # Whether Murmur enables draft windows by default (decided by decoder size,
    # paper Sec. 4.3): on for VibeVoice-ASR's 7B decoder, off for MOSS's 0.6B.
    draft_windows_default: bool = False
    # Merge consecutive same-speaker segments closer than this (seconds); None = no merge.
    merge_gap_s: float | None = None
    #: Keep segment times inside their chunk when stitching.
    clamp_to_chunk: bool = False
    #: The server-side drafter also uses the chunk's diarization (speaker-turn drafts).
    drafts_use_diarization: bool = False
    #: vLLM build without a custom-proposer hook: Murmur's drafter replaces the n-gram
    #: proposer, and this names the drafter ("vibevoice", "text"). None: custom proposer class.
    ngram_patch: str | None = None
    #: Longest audio the model can decode in one request (seconds); None = no known limit.
    #: A single pass over longer audio is split into the fewest pieces that fit.
    max_audio_s: float | None = None

    def __init__(self, url: str, served_name: str | None = None, timeout: float = 3600.0):
        self.url = url.rstrip("/")
        self.served_name = served_name or self.default_served_name()
        self.timeout = timeout

    def close(self) -> None:
        """Release resources owned by this client, if any."""

    def default_served_name(self) -> str:
        return self.hf_id

    def budget(self, audio: np.ndarray) -> int:
        raise NotImplementedError

    def _post(self, audio: np.ndarray, max_tokens: int, headers: dict) -> tuple[str, str | None]:
        raise NotImplementedError

    def decode(self, audio: np.ndarray, request_id: str | None = None, retries: int = 3) -> Reply:
        """Send one chunk (at ``sample_rate``) and wait for the transcript.

        ``request_id`` tags the request so the server-side drafter can find
        this chunk's draft file (``X-Request-Id: murmur_<id>``).
        """
        headers = {"X-Request-Id": f"murmur_{request_id}"} if request_id else {}
        t0 = time.perf_counter()
        for attempt in range(retries):
            try:
                text, finish = self._post(audio, self.budget(audio), headers)
                return Reply(text, time.perf_counter() - t0, finish)
            except requests.RequestException:
                if attempt == retries - 1:
                    return Reply("", time.perf_counter() - t0, None, failed=True)
                time.sleep(5)
        raise AssertionError("unreachable")

    def parse(self, text: str, keep_raw: bool = False) -> list[LocalSegment]:
        raise NotImplementedError

    def flagged(self, reply: Reply) -> bool:
        """True if the chunk failed: the request failed or the output never closed (repetition loop)."""
        raise NotImplementedError

    @classmethod
    def model_path(cls, vllm_bin: str) -> str:
        """What to pass to ``vllm serve`` (the Hugging Face id unless a model needs local setup)."""
        return cls.hf_id

    @classmethod
    def server_args(cls, batch_size: int, speculative_config: str | None, single_pass: bool = False,
                    gpu_memory_utilization: float | None = None) -> tuple[list[str], dict]:
        """``vllm serve`` arguments and extra environment variables for this model."""
        raise NotImplementedError


class DiarizingModel(ModelClient):
    """A model whose output has speaker labels and timestamps (VibeVoice-ASR, MOSS-Transcribe-Diarize).

    ``parse()`` returns the model's own segments; Murmur links speakers across chunks."""

    diarizes = True


class TranscriptModel(ModelClient):
    """A model that writes plain text (Voxtral).

    ``parse()`` returns the chunk's text as one segment without speaker or times; Murmur
    times each word by aligning it to the short-window ASR's words and assigns speakers
    from diarization (``murmur.attribute``)."""

    diarizes = False
    ngram_patch = "text"

    def parse(self, text: str, keep_raw: bool = False) -> list[LocalSegment]:
        text = text.strip()
        return [LocalSegment(0.0, None, "", text)] if text else []

    def flagged(self, reply: Reply) -> bool:
        return reply.failed or reply.finish == "length" or looped(reply.text)


def looped(text: str, tail: int = 120, max_period: int = 30, share: float = 0.9) -> bool:
    """A repetition loop: the last ``tail`` words repeat with some period of at most
    ``max_period`` words (``share`` of them equal the word one period earlier)."""
    w = re.sub(r"[^\w\s']", " ", text.lower()).split()
    if len(w) < tail:
        return False
    w = w[-tail:]
    return any(sum(w[i] == w[i - p] for i in range(p, tail)) >= share * (tail - p) for p in range(1, max_period + 1))
