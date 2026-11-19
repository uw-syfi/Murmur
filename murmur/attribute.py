"""Speakers and timestamps for a model that writes plain text (TranscriptModel).

1. Word times: the model's words are aligned to the short-window ASR's words of the same
   audio (which have timestamps) by a longest-matching-blocks alignment on normalized
   words; matched words take the ASR word's time, the rest are interpolated between
   their matched neighbours.
2. Speakers: each word takes the diarization speaker talking at its time (the nearest
   turn if none is).
3. Consecutive words of one speaker, with no gap longer than ``max_gap_s``, form a segment.

Speaker accuracy is therefore bounded by the diarization, as in WhisperX.
"""

from __future__ import annotations

import bisect
import re
from difflib import SequenceMatcher

from .types import Segment

WORD_S = 0.3  # assumed word duration where nothing else is known


def _norm(w: str) -> str:
    return re.sub(r"[^\w']", "", w.lower())


def word_times(words: list[str], ref: list[tuple[float, str]], start: float, end: float) -> list[float]:
    """A time (seconds) for each of ``words``, from ``ref`` = [(time, word), ...] of the same audio."""
    n = len(words)
    if n == 0:
        return []
    times: list[float | None] = [None] * n
    if ref:
        sm = SequenceMatcher(None, [_norm(w) for w in words], [_norm(w) for _, w in ref], autojunk=False)
        for blk in sm.get_matching_blocks():
            for k in range(blk.size):
                times[blk.a + k] = ref[blk.b + k][0]
    anchors = [i for i, t in enumerate(times) if t is not None]
    if not anchors:  # nothing matched: spread the words evenly over the audio
        step = (end - start) / n
        return [start + (i + 0.5) * step for i in range(n)]
    out: list[float] = [0.0] * n
    for i in range(n):
        if times[i] is not None:
            out[i] = times[i]
            continue
        j = bisect.bisect_left(anchors, i)
        lo, hi = (anchors[j - 1] if j > 0 else None), (anchors[j] if j < len(anchors) else None)
        if lo is not None and hi is not None:
            out[i] = times[lo] + (times[hi] - times[lo]) * (i - lo) / (hi - lo)
        elif hi is not None:
            out[i] = max(start, times[hi] - WORD_S * (hi - i))
        else:
            out[i] = min(end, times[lo] + WORD_S * (i - lo))
    for i in range(1, n):  # keep time order
        out[i] = max(out[i], out[i - 1])
    return out


def speaker_at(t: float, timeline: list[tuple[float, float, str]]) -> str:
    """The diarization speaker talking at ``t`` (the longest-running turn if several);
    the nearest turn's speaker if nobody is."""
    if not timeline:
        return "S0"
    on = [(b - a, spk) for a, b, spk in timeline if a <= t <= b]
    if on:
        return max(on)[1]
    return min(timeline, key=lambda x: min(abs(t - x[0]), abs(t - x[1])))[2]


def attribute(text: str, ref: list[tuple[float, str]], start: float, end: float,
              timeline: list[tuple[float, float, str]] | None, max_gap_s: float = 1.5) -> list[Segment]:
    """Segments for one chunk's plain-text output.

    ``ref``: the short-window ASR's (time, word) pairs for the chunk, times absolute;
    ``start``/``end``: the chunk in the recording; ``timeline``: diarization (absolute)."""
    words = text.split()
    if not words:
        return []
    ts = word_times(words, ref, start, end)
    tl = [t for t in (timeline or []) if t[1] > start - 5 and t[0] < end + 5]
    segs: list[Segment] = []
    for w, t in zip(words, ts):
        spk = speaker_at(t, tl)
        if segs and segs[-1].speaker == spk and t - segs[-1].end <= max_gap_s:
            s = segs[-1]
            segs[-1] = Segment(spk, s.start, max(s.end, t + WORD_S), f"{s.text} {w}")
        else:
            segs.append(Segment(spk, t, t + WORD_S, w))
    return segs
