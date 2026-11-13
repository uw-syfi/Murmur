"""CPU tests: output parsing, chunking, stitching and the drafter (no model, no GPU)."""

from murmur.audio import chunk_bounds
from murmur.drafting.vllm_proposer import DraftTranscript, VibeVoiceDrafter
from murmur.models import LocalSegment, MossClient, Reply, VibeVoiceClient, get_model
from murmur.stitch import chunk_speaker_count, link_speakers, stitch

VV = VibeVoiceClient("http://x")
MOSS = MossClient("http://x")


def test_vibevoice_parse():
    text = '[{"Start":0.0,"End":2.5,"Speaker":0,"Content":"hello there"},{"Start":2.5,"Speaker":1,"Content":"hi"}]'
    segs = VV.parse(text)
    assert [(s.start, s.end, s.speaker, s.text) for s in segs] == [(0.0, 2.5, "0", "hello there"),
                                                                   (2.5, None, "1", "hi")]
    assert not VV.flagged(Reply(text, 1.0, "stop"))


def test_vibevoice_loop_is_flagged_and_kept_raw_only_on_request():
    loop = '[{"Start":0.0,"End":2.5,"Speaker":0,"Content":"yeah ha ha ha ha ha ha ha'
    assert VV.flagged(Reply(loop, 1.0, "length"))
    assert VV.parse(loop) == []
    assert VV.parse(loop, keep_raw=True)[0].text == loop


def test_moss_parse_and_flag():
    text = "[0.00][S01]hello there[1.84][2.10][S02]hi[2.90]"
    segs = MOSS.parse(text)
    assert [(s.start, s.end, s.speaker, s.text) for s in segs] == [(0.0, 1.84, "S01", "hello there"),
                                                                   (2.1, 2.9, "S02", "hi")]
    assert not MOSS.flagged(Reply(text, 1.0))
    assert MOSS.flagged(Reply("[0.00][S01]ha ha ha ha", 1.0))


def test_model_names():
    assert get_model("vibevoice") is VibeVoiceClient and get_model("MOSS") is MossClient


def test_chunk_bounds_follow_speech_regions():
    tl = [(0, 100, "A"), (110, 300, "B"), (320, 400, "A"), (410, 700, "B")]
    assert chunk_bounds(700, 330, tl) == [(0, 300), (320, 400), (410, 700)]
    assert chunk_bounds(700, 330, None) == [(0, 330), (330, 660), (660, 700)]


def test_link_speakers_is_one_to_one():
    parsed = [(0, 10, "0"), (10, 20, "1")]
    tl = [(0, 20, "SPK_A")]  # diarization under-counts: one speaker
    m = link_speakers(parsed, tl, (0, 20))
    assert sorted(m.values()) == ["SPK_A", "SPK_A"]  # leftover joins its best overlap
    tl2 = [(0, 10, "SPK_A"), (10, 20, "SPK_B")]
    assert link_speakers(parsed, tl2, (0, 20)) == {"0": "SPK_A", "1": "SPK_B"}


def test_stitch_offsets_and_links_across_chunks():
    pieces = [(0.0, 20.0, [LocalSegment(0, 9, "0", "a"), LocalSegment(10, 19, "1", "b")]),
              (30.0, 50.0, [LocalSegment(0, 9, "0", "c"), LocalSegment(10, 19, "1", "d")])]
    tl = [(0, 10, "A"), (10, 20, "B"), (30, 40, "B"), (40, 50, "A")]
    segs = stitch(pieces, tl)
    assert [(s.speaker, s.start, s.text) for s in segs] == [("A", 0, "a"), ("B", 10, "b"), ("B", 30, "c"), ("A", 40, "d")]


def test_stitch_without_diarization_prefixes_chunks_and_merges():
    pieces = [(0.0, 20.0, [LocalSegment(0, 5, "0", "a"), LocalSegment(5.5, 8, "0", "b")]),
              (20.0, 40.0, [LocalSegment(0, 5, "0", "c")])]
    segs = stitch(pieces, None, merge_gap_s=1.5)
    assert [(s.speaker, s.text) for s in segs] == [("c000:0", "a b"), ("c001:0", "c")]


def test_chunk_speaker_count_ignores_brief_speakers():
    chunk = [LocalSegment(0, 50, "0", "x"), LocalSegment(50, 100, "1", "y"), LocalSegment(100, 101, "2", "z")]
    assert chunk_speaker_count([chunk]) == 2


class CharTok:
    """A one-token-per-character tokenizer, enough to drive the drafter."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(map(chr, ids))


def test_vibevoice_drafter_proposes_end_as_next_start_and_transcript_words():
    tok = CharTok()
    words = [(0.0, "hello"), (0.5, "there"), (1.0, "general"), (1.6, "kenobi")]
    d = VibeVoiceDrafter(DraftTranscript(words, tok), tok, "en")
    gen = tok.encode('[{"Start":0.0,"End":1.9,"Speaker":0,"Content":"hello there"},{"Start":')
    assert tok.decode(d.propose(gen, 8)) == '1.9,"End'  # structure draft: next start = previous end
    d = VibeVoiceDrafter(DraftTranscript(words, tok), tok, "en")  # the drafter is incremental: new request
    gen = tok.encode('[{"Start":0.0,"End":1.9,"Speaker":0,"Content":"hello there gen')
    assert tok.decode(d.propose(gen, 6)) == "eral k"  # text draft from the transcript
