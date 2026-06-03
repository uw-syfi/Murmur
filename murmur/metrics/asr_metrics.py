"""ASR evaluation metrics: WER, CER, cpWER, tcpWER, DER.

Each output segment is a plain dict with these keys (both snake_case and Title Case
spellings are accepted everywhere):
    {
        "Speaker ID"  / "speaker_id"  : str,
        "Start time"  / "start_time"  : float | "HH:MM:SS.mmm",
        "End time"    / "end_time"    : float | "HH:MM:SS.mmm",
        "Content"     / "text"        : str,
    }
(for VibeVoice ASR)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from jiwer import wer as jiwer_wer

log = logging.getLogger(__name__)


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class MetricsResult:
    wer: float = math.nan
    cer: float = math.nan
    cpwer: float = math.nan
    tcpwer: float = math.nan
    der: float = math.nan

    def as_dict(self) -> dict[str, float]:
        return {
            "wer": self.wer,
            "cer": self.cer,
            "cpwer": self.cpwer,
            "tcpwer": self.tcpwer,
            "der": self.der,
        }

    def __str__(self) -> str:
        pairs = [
            ("WER", self.wer),
            ("CER", self.cer),
            ("cpWER", self.cpwer),
            ("tcpWER", self.tcpwer),
            ("DER", self.der),
        ]
        parts = [f"{name}={val:.3f}%" for name, val in pairs if not math.isnan(val)]
        return " | ".join(parts) if parts else "(no metrics computed)"


# ── Timestamp helper ──────────────────────────────────────────────────────────

def _parse_ts(ts) -> float:
    """Parse ``HH:MM:SS.mmm`` or a numeric value to seconds."""
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, str):
        parts = ts.split(":")
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        try:
            return float(ts)
        except ValueError:
            return 0.0
    return 0.0


def _seg_text(seg: dict) -> str:
    return seg.get("Content", seg.get("text", ""))


def _seg_speaker(seg: dict) -> str:
    return seg.get("Speaker ID", seg.get("speaker_id", "unknown"))


def _seg_start(seg: dict) -> float:
    return _parse_ts(seg.get("Start time", seg.get("start_time", 0.0)))


def _seg_end(seg: dict) -> float:
    return _parse_ts(seg.get("End time", seg.get("end_time", 0.0)))


# ── meeteval STM conversion ───────────────────────────────────────────────────

def _to_stm(segments: list[dict], meeting_id: str):
    from meeteval.io import STM, STMLine

    lines = [
        STMLine(
            filename=meeting_id,
            channel=0,
            speaker_id=_seg_speaker(seg),
            begin_time=_seg_start(seg),
            end_time=_seg_end(seg),
            transcript=_seg_text(seg),
        )
        for seg in segments
    ]
    return STM(lines)


# ── Pure-Python Levenshtein (no jiwer) ───────────────────────────────────────

def _levenshtein(a: list, b: list) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


# ── WER ───────────────────────────────────────────────────────────────────────

def compute_wer(ref: str, hyp: str) -> float:
    if not ref.strip():
        return float("nan")
    return round(jiwer_wer([ref], [hyp or ""]) * 100, 3)

# ── CER ───────────────────────────────────────────────────────────────────────

def compute_cer(ref: str, hyp: str) -> float:
    """Character Error Rate (%).

    Uses *jiwer* when available, falls back to pure-Python Levenshtein.
    """
    if not ref.strip():
        return math.nan
    try:
        from jiwer import cer as _cer

        return round(_cer([ref], [hyp or ""]) * 100, 3)
    except ImportError:
        ref_chars = list(ref)
        hyp_chars = list(hyp or "")
        d = _levenshtein(ref_chars, hyp_chars)
        return round(d / len(ref_chars) * 100, 3)


# ── cpWER ─────────────────────────────────────────────────────────────────────

def compute_cpwer(
    ref_segments: list[dict],
    hyp_segments: list[dict],
    meeting_id: str = "meeting",
) -> float:
    """Concatenated minimum-permutation WER (speaker-aware, %).
    """
    if not ref_segments or not hyp_segments:
        return math.nan
    try:
        from meeteval.wer.api import cpwer

        ref_stm = _to_stm(ref_segments, meeting_id)
        hyp_stm = _to_stm(hyp_segments, meeting_id)

        if len(ref_stm) == 0 or len(hyp_stm) == 0:
            return float("nan")

        try:
            result = cpwer(ref_stm, hyp_stm)
            total_errors = sum(v.errors for v in result.values())
            total_length = sum(v.length for v in result.values())
            if total_length == 0:
                return float("nan")
            return round(total_errors / total_length * 100, 3)
        except Exception as e:
            log.warning(f"cpWER computation failed: {e}")
            return float("nan")
        
    except ImportError:
        ref_text = " ".join(_seg_text(s) for s in ref_segments)
        hyp_text = " ".join(_seg_text(s) for s in hyp_segments)
        return compute_wer(ref_text, hyp_text)


# ── tcpWER ────────────────────────────────────────────────────────────────────

def compute_tcpwer(
    ref_segments: list[dict],
    hyp_segments: list[dict],
    collar: float = 5.0,
    meeting_id: str = "meeting",
) -> float:
    """Time-constrained minimum-permutation WER (%).

    Matches hypothesis segments to reference within ±*collar* seconds.
    """
    if not ref_segments or not hyp_segments:
        return math.nan
    try:
        from meeteval.wer.api import tcpwer

        result = tcpwer(
            _to_stm(ref_segments, meeting_id),
            _to_stm(hyp_segments, meeting_id),
            collar=collar,
        )
        total_errors = sum(v.errors for v in result.values())
        total_length = sum(v.length for v in result.values())
        if total_length == 0:
            return math.nan
        return round(total_errors / total_length * 100, 3)
    except ImportError:
        return compute_cpwer(ref_segments, hyp_segments, meeting_id)


# ── DER ───────────────────────────────────────────────────────────────────────

def compute_der(
    ref_rttm_segments: list[tuple[float, float, str]],
    hyp_segments: list[dict],
    collar: float = 0.25,
) -> float:
    if not ref_rttm_segments or not hyp_segments:
        return math.nan
    try:
        from pyannote.core import Annotation, Segment
        from pyannote.metrics.diarization import DiarizationErrorRate

        ref_ann = Annotation()
        for start, end_s, spk in ref_rttm_segments:
            ref_ann[Segment(start, end_s)] = spk

        hyp_ann = Annotation()
        for s in hyp_segments:
            hyp_ann[Segment(_seg_start(s), _seg_end(s))] = _seg_speaker(s)

        metric = DiarizationErrorRate(collar=collar, skip_overlap=False)
        der = metric(ref_ann, hyp_ann)
        return round(der * 100, 3)
    except ImportError:
        log.warning(
            "pyannote.metrics not installed — DER will be NaN. "
            "pip install pyannote.metrics"
        )
        return math.nan


def evaluate(
    ref_segments: list[dict],
    hyp_segments: list[dict],
    ref_rttm: list[tuple[float, float, str]] | None = None,
    meeting_id: str = "meeting",
) -> MetricsResult:
    """Compute all available metrics in one call.

    Parameters
    ----------
    ref_segments:
        Reference transcript segments.
    hyp_segments:
        Hypothesis (system output) transcript segments.
    ref_rttm:
        Optional list of ``(start_s, end_s, speaker)`` tuples for DER.
        If not provided DER is left as NaN.
    meeting_id:
        Meeting identifier used in STM conversion (for meeteval).

    Returns
    -------
    MetricsResult
        All metrics.  Any metric whose dependency is missing is NaN.
    """
    ref_text = " ".join(_seg_text(s) for s in ref_segments)
    hyp_text = " ".join(_seg_text(s) for s in hyp_segments)

    return MetricsResult(
        wer=compute_wer(ref_text, hyp_text),
        cer=compute_cer(ref_text, hyp_text),
        cpwer=compute_cpwer(ref_segments, hyp_segments, meeting_id),
        tcpwer=compute_tcpwer(ref_segments, hyp_segments, meeting_id=meeting_id),
        der=compute_der(ref_rttm, hyp_segments) if ref_rttm is not None else math.nan,
    )
