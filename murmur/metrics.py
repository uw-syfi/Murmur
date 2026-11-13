"""Scoring as in the paper: WER (CER for Mandarin) with jiwer, cpWER and tcpWER (5 s collar)
with MeetEval, DER with pyannote.metrics (0.25 s collar, overlap scored).

Mandarin is scored per character: a space is put between CJK characters before diffing,
so the word-level scorers count characters, while Latin and digit runs stay single tokens.
"""

from __future__ import annotations

import logging
import math
import re
import unicodedata

log = logging.getLogger(__name__)

_FILLERS = re.compile(r"\b(uh+|um+|mm+|hmm+|hm+|mhm)\b")
_NONSPEECH = re.compile(r"\[[^\]]*\]|<[^>]*>")  # [laughter], <noise>, ...
_CJK_RANGES = ((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF), (0x3040, 0x30FF))


def normalize(text: str) -> str:
    """Text normalization applied to references and hypotheses before scoring."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = _NONSPEECH.sub("", text)
    text = _FILLERS.sub("", text)
    text = re.sub(r"[^\w\s']", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def cjk_char_split(text: str) -> str:
    """Put spaces between CJK characters so word-level scorers count characters."""
    out, prev = [], False
    for ch in text:
        cur = _is_cjk(ch)
        if cur or prev:
            out.append(" ")
        out.append(ch)
        prev = cur
    return re.sub(r"\s+", " ", "".join(out)).strip()


def _cjk_normalizer(seg: dict) -> dict:
    seg["words"] = cjk_char_split(seg["words"])
    return seg


def _to_stm(segments: list[dict], meeting_id: str):
    from meeteval.io import STM, STMLine
    return STM([STMLine(filename=meeting_id, channel=0, speaker_id=s["Speaker ID"], begin_time=s["Start time"],
                        end_time=max(s["End time"], s["Start time"]),  # meeteval rejects end < start
                        transcript=s["Content"])
                for s in segments])


def _meeteval_rate(fn, ref_segments, hyp_segments, meeting_id, cjk, **kw) -> float:
    """cpWER / tcpWER in percent. No hypothesis scores 100 (all deletions), not NaN, so a
    failed recording counts against the mean instead of dropping out of it."""
    if not ref_segments:
        return math.nan
    if not hyp_segments:
        return 100.0
    try:
        res = fn(_to_stm(ref_segments, meeting_id), _to_stm(hyp_segments, meeting_id),
                 normalizer=_cjk_normalizer if cjk else None, **kw)
    except Exception as e:  # e.g. meeteval's limit on the number of speakers to permute
        log.warning("%s failed: %s", fn.__name__, e)
        return math.nan
    length = sum(v.length for v in res.values())
    return round(sum(v.errors for v in res.values()) / length * 100, 3) if length else math.nan


def compute_cpwer(ref_segments, hyp_segments, meeting_id="meeting", cjk=False) -> float:
    from meeteval.wer.api import cpwer
    return _meeteval_rate(cpwer, ref_segments, hyp_segments, meeting_id, cjk)


def compute_tcpwer(ref_segments, hyp_segments, collar=5.0, meeting_id="meeting", cjk=False) -> float:
    from meeteval.wer.api import tcpwer
    return _meeteval_rate(tcpwer, ref_segments, hyp_segments, meeting_id, cjk, collar=collar)


def compute_der(ref_rttm: list[tuple[float, float, str]], hyp_segments: list[dict], collar=0.25) -> float:
    if not ref_rttm:
        return math.nan
    if not hyp_segments:
        return 100.0
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate
    ref, hyp = Annotation(), Annotation()
    for a, b, spk in ref_rttm:
        ref[Segment(a, b)] = spk
    for s in hyp_segments:
        hyp[Segment(s["Start time"], s["End time"])] = s["Speaker ID"]
    return round(DiarizationErrorRate(collar=collar, skip_overlap=False)(ref, hyp) * 100, 3)


def evaluate(hyp, ref_segments: list[dict], ref_rttm: list[tuple[float, float, str]] | None = None,
             meeting_id: str = "meeting", cjk: bool = False) -> dict:
    """Score one recording: ``hyp`` is a Transcript or a list of Segments.

    ``ref_segments`` are dicts with "Speaker ID", "Start time", "End time" and "Content"
    (already normalized); ``ref_rttm`` is [(start, end, speaker), ...] for DER.
    Returns {"wer", "cer", "cpwer", "tcpwer", "der"} in percent (NaN when not computable).
    """
    import jiwer

    segs = getattr(hyp, "segments", hyp)
    ref_text = " ".join(s["Content"] for s in ref_segments)
    hyp_text = normalize(" ".join(s.text for s in segs))
    hyp_dicts = [{"Speaker ID": s.speaker, "Start time": s.start, "End time": s.end, "Content": normalize(s.text)}
                 for s in segs]
    if ref_text.strip():
        r, h = (cjk_char_split(ref_text), cjk_char_split(hyp_text)) if cjk else (ref_text, hyp_text)
        wer = round(jiwer.wer(r, h) * 100, 3)
        cer = round(jiwer.cer(ref_text, hyp_text) * 100, 3)
    else:
        wer = cer = math.nan
    if ref_segments and any(s["Speaker ID"] for s in ref_segments):
        cpwer = compute_cpwer(ref_segments, hyp_dicts, meeting_id=meeting_id, cjk=cjk)
        tcpwer = compute_tcpwer(ref_segments, hyp_dicts, meeting_id=meeting_id, cjk=cjk)
        der = compute_der(ref_rttm, hyp_dicts) if ref_rttm else math.nan
    else:
        cpwer = tcpwer = der = math.nan
    return {"wer": wer, "cer": cer, "cpwer": cpwer, "tcpwer": tcpwer, "der": der}
