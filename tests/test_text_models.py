"""CPU tests for models that write plain text (Voxtral): word times, speakers, drafter, server flags."""

import numpy as np

from murmur.api import Murmur
from murmur.audio import max_length_bounds
from murmur.attribute import attribute, speaker_at, word_times
from murmur.drafting.vllm_proposer import DraftTranscript, MurmurCore, TextDrafter
from murmur.models import Reply, VoxtralClient, get_model
from murmur.server import server_command

SR = 16_000


def test_word_times_take_matched_times_and_interpolate_the_rest():
    ref = [(1.0, "okay"), (1.5, "so"), (2.0, "we"), (3.0, "start")]
    ts = word_times("Okay, so um we start".split(), ref, 0.0, 10.0)
    assert ts[0] == 1.0 and ts[1] == 1.5 and ts[3] == 2.0 and ts[4] == 3.0
    assert 1.5 < ts[2] < 2.0  # "um" is not in the reference: between its neighbours
    assert word_times(["a", "b"], [], 0.0, 4.0) == [1.0, 3.0]  # nothing to align: spread evenly


def test_attribute_assigns_the_diarization_speaker_and_groups_words():
    ref = [(0.5, "hello"), (1.0, "there"), (5.0, "hi"), (5.5, "back")]
    tl = [(0.0, 2.0, "A"), (4.5, 6.0, "B")]
    segs = attribute("Hello there. Hi back.", ref, 0.0, 10.0, tl)
    assert [(s.speaker, s.text) for s in segs] == [("A", "Hello there."), ("B", "Hi back.")]
    assert segs[0].start == 0.5 and segs[1].start == 5.0
    assert speaker_at(3.0, tl) == "A"  # nobody talking: the nearest turn


def test_text_drafter_follows_the_transcript_and_copies_loops():
    from test_pipeline import CharTok
    tok = CharTok()
    words = [(0.0, "okay"), (0.4, "so"), (0.8, "we"), (1.2, "start")]
    d = TextDrafter(DraftTranscript(words, tok), tok)
    assert tok.decode(d.propose(tok.encode("ok"), 6)) == "ay so "  # the transcript's first words
    d = TextDrafter(DraftTranscript(words, tok), tok)
    assert tok.decode(d.propose(tok.encode("Okay, so we st"), 3)) == "art"  # n-gram lookup
    d = TextDrafter(DraftTranscript(words, tok), tok)
    assert tok.decode(d.propose(tok.encode("yeah yeah yeah yeah"), 5)) == " yeah"  # loop copy


def test_draft_file_found_without_the_transcription_endpoint_suffix(tmp_path):
    core = object.__new__(MurmurCore)
    core.dir, core.kind = tmp_path, "text"
    assert core._draft_path("transcribe-murmur_ab12_3_0") == tmp_path / "ab12_3.json"  # not written yet
    (tmp_path / "ab12_3.json").write_text("{}")
    assert core._draft_path("transcribe-murmur_ab12_3_0") == tmp_path / "ab12_3.json"


def test_voxtral_client_parse_flag_and_server():
    assert get_model("voxtral") is VoxtralClient and not VoxtralClient.diarizes
    c = VoxtralClient("http://x")
    assert [(s.speaker, s.text) for s in c.parse(" hello there ")] == [("", "hello there")]
    assert c.flagged(Reply("we start. " + "yeah " * 130, 1.0)) and not c.flagged(Reply("we start now", 1.0))
    assert c.flagged(Reply("ok " + "and then we will go to the next slide " * 20, 1.0))  # a long phrase loops too
    assert not c.flagged(Reply(" ".join(f"w{i}" for i in range(300)), 1.0))
    cmd, env = server_command(VoxtralClient, port=1, speculative="murmur", draft_dir="/d", vllm_bin="vllm",
                              prepare_model=False)
    assert '"method":"ngram"' in cmd[cmd.index("--speculative-config") + 1]
    assert env["MURMUR_MODEL"] == "text" and env["MURMUR_LONG_AUDIO"] == "1" and "_vllm_site" in env["PYTHONPATH"]
    cmd, env = server_command(VoxtralClient, port=1, speculative=None, vllm_bin="vllm", prepare_model=False)
    assert "MURMUR_MODEL" not in env and "_vllm_site" in env["PYTHONPATH"]  # long audio still one request


class FakeVoxtral(VoxtralClient):
    def __init__(self):
        super().__init__("http://fake")
        self.calls = []

    def decode(self, audio, request_id=None, retries=3):
        self.calls.append(len(audio) / self.sample_rate)
        return Reply("Hello there. Hi back.", 0.01)


def fake_asr(wav16):
    """Short-window ASR stand-in: the same four words in every chunk, times relative to it."""
    return [(0.5, "Hello"), (1.0, "there."), (5.0, "Hi"), (5.5, "back.")]


def test_text_model_gets_speakers_and_times_from_asr_and_diarization():
    client = FakeVoxtral()
    m = Murmur(client, chunk_s=20.0, drafter=fake_asr)
    tl = [(0, 2, "A"), (4.5, 20, "B"), (25, 27, "B"), (29.5, 45, "A")]
    t = m.transcribe(np.zeros(SR * 50, dtype=np.float32), sr=SR, timeline=tl)
    assert len(client.calls) == 2
    assert [(s.speaker, s.start, s.text) for s in t.segments] == [
        ("A", 0.5, "Hello there."), ("B", 5.0, "Hi back."), ("B", 25.5, "Hello there."), ("A", 30.0, "Hi back.")]
    t = m.transcribe(np.zeros(SR * 50, dtype=np.float32), sr=SR, timeline=tl, single_pass=True)
    assert len(client.calls) == 3 and round(client.calls[-1]) == 50  # one request, still attributed
    assert [s.speaker for s in t.segments] == ["A", "B"]


def test_single_pass_over_the_model_limit_is_split_into_the_fewest_pieces():
    assert max_length_bounds(1000, 1800, None) == [(0.0, 1000)]
    assert max_length_bounds(3000, 1800, None) == [(0.0, 1500.0), (1500.0, 3000)]
    tl = [(t, t + 50, "A") for t in range(0, 3000, 60)]  # speech every minute
    b = max_length_bounds(3000, 1800, tl)
    assert len(b) == 2 and all(e - s <= 1800 for s, e in b)
    client = FakeVoxtral()
    client.max_audio_s = 30.0
    Murmur(client, drafter=fake_asr).transcribe(np.zeros(SR * 50, dtype=np.float32), sr=SR,
                                                timeline=[(0, 20, "A"), (26, 50, "B")], single_pass=True)
    assert [round(d) for d in client.calls] == [20, 24]  # two pieces under 30 s, cut at the pause


def test_voxtral_small_uses_the_voxtral_vllm(monkeypatch):
    from murmur.server import find_vllm
    monkeypatch.setenv("MURMUR_VLLM_VOXTRAL", "/envs/vibevoice-vllm/bin/vllm")
    assert find_vllm("voxtral-small") == "/envs/vibevoice-vllm/bin/vllm"
