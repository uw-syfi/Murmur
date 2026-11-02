"""Stitching chunk outputs into one transcript (paper Sec. 4.4).

Timestamps are shifted by each chunk's start time. Speakers keep the model's own
labels within a chunk; diarization only links them across chunks, by a one-to-one
(Hungarian) assignment that maximizes temporal overlap.
"""

from __future__ import annotations

from .models.base import LocalSegment
from .types import Segment

# A speaker counts toward the speaker-count hint only if it holds at least this
# share of a chunk's speech (models sometimes emit brief spurious speakers).
SPEAKER_HINT_MIN_SHARE = 0.10

Piece = tuple[float, float, list[LocalSegment]]  # (chunk_start, chunk_end, model segments)


def chunk_speaker_count(chunks: list[list[LocalSegment]], min_share: float = SPEAKER_HINT_MIN_SHARE) -> int:
    """Largest number of speakers the model reports within any one chunk."""
    best = 0
    for segs in chunks:
        talk: dict[str, float] = {}
        for s in segs:
            if not s.text or s.speaker == "unknown":
                continue
            end = s.start if s.end is None else s.end
            talk[s.speaker] = talk.get(s.speaker, 0.0) + max(0.0, end - s.start)
        tot = sum(talk.values())
        n = sum(1 for v in talk.values() if v >= min_share * tot) if tot > 0 else len(talk)
        best = max(best, n)
    return best


def link_speakers(parsed: list[tuple[float, float, str]], timeline, window: tuple[float, float]) -> dict[str, str]:
    """Map each local speaker of one chunk to a diarization speaker (one-to-one by overlap).

    Leftover local speakers (more than diarization speakers in this window) join the
    diarization speaker they overlap most.
    """
    from scipy.optimize import linear_sum_assignment

    ws, we = window
    tl = [(t0, t1, g) for t0, t1, g in timeline if t1 > ws and t0 < we]
    overlap: dict[str, dict[str, float]] = {}
    for a, b, loc in parsed:
        bucket = overlap.setdefault(loc, {})
        for t0, t1, g in tl:
            ov = min(b, t1) - max(a, t0)
            if ov > 0:
                bucket[g] = bucket.get(g, 0.0) + ov
    locs = sorted(overlap)
    globs = sorted({g for ov in overlap.values() for g in ov})
    mapping: dict[str, str] = {}
    if locs and globs:
        cost = [[-overlap[l].get(g, 0.0) for g in globs] for l in locs]
        rows, cols = linear_sum_assignment(cost)
        for i, j in zip(rows, cols):
            if cost[i][j] < 0:
                mapping[locs[i]] = globs[j]
    for l in locs:
        if l not in mapping:
            mapping[l] = max(overlap[l], key=overlap[l].get) if overlap[l] else "unknown"
    return mapping


def _merge(segs: list[Segment], gap: float) -> list[Segment]:
    out: list[Segment] = []
    for s in segs:
        if out and s.speaker == out[-1].speaker and s.start - out[-1].end <= gap:
            p = out[-1]
            out[-1] = Segment(p.speaker, p.start, s.end, f"{p.text} {s.text}".strip())
        else:
            out.append(s)
    return out


def stitch(pieces: list[Piece], timeline=None, merge_gap_s: float | None = None,
           clamp: bool = True) -> list[Segment]:
    """Global segments from per-chunk model output.

    ``timeline`` (diarization) links speakers across chunks; without it, speaker labels
    are prefixed with the chunk index (``c003:S01``) unless there is a single chunk.
    ``clamp`` keeps segment times inside their chunk (VibeVoice-ASR); MOSS output is
    shifted without clamping, as in the paper's runs.
    """
    out: list[Segment] = []
    for ci, (cs, ce, segs) in enumerate(pieces):
        rows = []
        for s in segs:
            a, b = cs + s.start, (ce if s.end is None else cs + s.end)
            if clamp:
                a, b = max(cs, min(a, ce)), max(cs, min(b, ce))
            rows.append((a, b, s.speaker, s.text))
        if timeline:
            mapping = link_speakers([(a, b, sp) for a, b, sp, _ in rows], timeline, (cs, ce))
        elif len(pieces) > 1:
            mapping = {sp: f"c{ci:03d}:{sp}" for _, _, sp, _ in rows}
        else:
            mapping = {sp: sp for _, _, sp, _ in rows}
        out += [Segment(mapping.get(sp, "unknown"), a, b, t) for a, b, sp, t in rows if t]
    out.sort(key=lambda s: s.start)
    return _merge(out, merge_gap_s) if merge_gap_s is not None else out
