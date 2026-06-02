"""Tests for the text/parse helpers in benchmarks/benchmark.py (CPU, no model)."""
import json

import pytest

from benchmark import normalize, format_ts, load_layer_ranges, _parse_stm, _parse_rttm


def test_normalize_lowercases_and_strips_fillers_and_annotations():
    assert normalize("Hello, [Laughter] uh WORLD!!") == "hello world"
    assert normalize("<noise> hmm test") == "test"
    assert normalize("  multiple   spaces ") == "multiple spaces"


def test_format_ts():
    assert format_ts(3661.5) == "1:01:01.500"
    assert format_ts(75.25) == "1:15.250"
    assert format_ts(-3) == "0:00.000"  # clamped to 0


def test_parse_stm(tmp_path):
    stm = tmp_path / "m.stm"
    stm.write_text(
        ";; a comment line\n"
        "file 1 spkA 0.0 2.5 <o,f0,male> hello world\n"
        "file 1 spkB 3.0 4.0 <o,f0,female> IGNORE_TIME_SEGMENT_IN_SCORING\n"
    )
    segs = _parse_stm(stm)
    assert len(segs) == 1
    assert segs[0]["Speaker ID"] == "spkA"
    assert segs[0]["Start time"] == 0.0
    assert segs[0]["End time"] == 2.5
    assert segs[0]["Content"] == "hello world"


def test_parse_rttm(tmp_path):
    rttm = tmp_path / "m.rttm"
    rttm.write_text("SPEAKER file 1 0.5 2.0 <NA> <NA> spk1 <NA> <NA>\n")
    segs = _parse_rttm(rttm)
    assert segs == [(0.5, 2.5, "spk1")]  # (start, start+duration, speaker)


def test_load_layer_ranges_maps_short_keys(tmp_path):
    cfg = tmp_path / "layers.json"
    cfg.write_text(json.dumps([
        {"start": 0, "end": 4, "sink": 16, "window": 256, "delay": 8, "rate": 4},
    ]))
    ranges = load_layer_ranges(str(cfg))
    assert ranges == [{
        "start": 0, "end": 4,
        "speech_sink": 16, "speech_window": 256,
        "slide_delay": 8, "slide_rate": 4,
    }]


def test_load_layer_ranges_rejects_unknown_key(tmp_path):
    cfg = tmp_path / "bad.json"
    cfg.write_text(json.dumps([{"start": 0, "end": 4, "bogus": 1}]))
    with pytest.raises(ValueError):
        load_layer_ranges(str(cfg))
