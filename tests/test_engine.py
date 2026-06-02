"""Pure-logic tests for murmur.inference.local_engine helpers (CPU, no model)."""
from murmur.inference.local_engine import (
    _parse_ts,
    _merge_consecutive,
    _dominant_speaker,
    _build_speaker_mapping,
    TranscriptSegment,
)


def test_parse_ts():
    assert _parse_ts(12.5) == 12.5
    assert _parse_ts("00:01:30.5") == 90.5
    assert _parse_ts("90") == 90.0
    assert _parse_ts("bad") == 0.0


def test_merge_consecutive_same_speaker_within_gap():
    segs = [
        TranscriptSegment("A", 0.0, 1.0, "hi"),
        TranscriptSegment("A", 1.5, 2.0, "there"),   # gap 0.5 <= 1.5 -> merge
        TranscriptSegment("B", 2.0, 3.0, "yo"),       # different speaker
        TranscriptSegment("A", 10.0, 11.0, "late"),   # gap 7.0 > 1.5 -> no merge
    ]
    merged = _merge_consecutive(segs, max_gap_s=1.5)
    assert [(s.speaker_id, s.start_s, s.end_s, s.text) for s in merged] == [
        ("A", 0.0, 2.0, "hi there"),
        ("B", 2.0, 3.0, "yo"),
        ("A", 10.0, 11.0, "late"),
    ]


def test_merge_consecutive_empty():
    assert _merge_consecutive([]) == []


def test_dominant_speaker_by_overlap():
    timeline = [(0.0, 5.0, "A"), (4.0, 10.0, "B")]
    assert _dominant_speaker(0.0, 3.0, timeline) == "A"
    assert _dominant_speaker(6.0, 9.0, timeline) == "B"
    assert _dominant_speaker(20.0, 30.0, timeline) == "unknown"  # no overlap


def test_build_speaker_mapping_majority_overlap():
    timeline = [(0.0, 5.0, "A"), (4.0, 10.0, "B")]
    parsed = [(0.0, 3.0, "0"), (6.0, 9.0, "1")]
    assert _build_speaker_mapping(parsed, timeline) == {"0": "A", "1": "B"}


def test_build_speaker_mapping_window_restricts_timeline():
    timeline = [(0.0, 5.0, "A"), (100.0, 105.0, "Z")]
    parsed = [(1.0, 4.0, "0")]
    mapping = _build_speaker_mapping(parsed, timeline, window_start=0.0, window_end=10.0)
    assert mapping == {"0": "A"}
