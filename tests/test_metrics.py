"""Pure-logic tests for murmur.metrics.asr_metrics (CPU, no model)."""
import math

from murmur.metrics.asr_metrics import (
    _parse_ts,
    _levenshtein,
    _seg_speaker,
    _seg_start,
    compute_wer,
    compute_cer,
    MetricsResult,
)


def test_parse_ts_numeric_and_hms():
    assert _parse_ts(12.5) == 12.5
    assert _parse_ts("01:00:00") == 3600.0
    assert _parse_ts("00:01:30.5") == 90.5
    assert _parse_ts("5.5") == 5.5
    assert _parse_ts("garbage") == 0.0


def test_levenshtein():
    assert _levenshtein(list("kitten"), list("sitting")) == 3
    assert _levenshtein(list("abc"), list("abc")) == 0
    assert _levenshtein([], list("abc")) == 3


def test_segment_key_aliases():
    assert _seg_speaker({"Speaker ID": "X"}) == "X"
    assert _seg_speaker({"speaker_id": "y"}) == "y"
    assert _seg_speaker({}) == "unknown"
    assert _seg_start({"start_time": "0:00:02"}) == 2.0


def test_wer_identical_and_half():
    assert compute_wer("hello world", "hello world") == 0.0
    assert compute_wer("hello world", "hello there") == 50.0


def test_wer_empty_ref_is_nan():
    assert math.isnan(compute_wer("", "anything"))


def test_cer():
    assert compute_cer("abc", "abc") == 0.0
    assert compute_cer("abc", "abd") == 33.333


def test_metrics_result_str_skips_nan():
    s = str(MetricsResult(wer=12.5, cer=3.0))
    assert "WER=12.500%" in s and "CER=3.000%" in s
    assert "cpWER" not in s  # NaN metrics are omitted
    assert str(MetricsResult()) == "(no metrics computed)"
