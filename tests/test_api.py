"""CPU tests for the pipeline, server flags and scoring (a fake model; no GPU, no vLLM)."""

import math

import numpy as np
import pytest

from murmur.api import Murmur
from murmur.models import MossClient, Reply, VibeVoiceClient
from murmur.server import server_command
from murmur.types import Segment

SR = 16_000


class FakeVibeVoice(VibeVoiceClient):
    """Answers every chunk with two speakers, without a server."""

    def __init__(self):
        super().__init__("http://fake")
        self.calls = []

    def decode(self, audio, request_id=None, retries=3):
        self.calls.append((len(audio) / self.sample_rate, request_id))
        return Reply('[{"Start":0.0,"End":4.0,"Speaker":0,"Content":"hello world"},'
                     '{"Start":5.0,"End":9.0,"Speaker":1,"Content":"good morning"}]', 0.01, "stop")


def test_transcribe_chunks_decodes_and_stitches():
    client = FakeVibeVoice()
    m = Murmur(client, chunk_s=20.0, batch_size=4)
    audio = np.zeros(SR * 50, dtype=np.float32)
    tl = [(0, 4.5, "A"), (4.5, 20, "B"), (25, 29.5, "B"), (29.5, 45, "A")]  # speakers swap in chunk 2
    t = m.transcribe(audio, sr=SR, timeline=tl)
    assert t.stats["n_chunks"] == 2 and len(client.calls) == 2
    assert [round(d, 1) for d, _ in client.calls] == [20.0, 20.0]  # chunks cut between speech regions
    assert [(s.speaker, s.start) for s in t.segments] == [("A", 0.0), ("B", 5.0), ("B", 25.0), ("A", 30.0)]
    assert t.text == "hello world good morning hello world good morning"


def test_single_pass_sends_the_whole_recording():
    client = FakeVibeVoice()
    t = Murmur(client).transcribe(np.zeros(SR * 50, dtype=np.float32), sr=SR, single_pass=True)
    assert len(client.calls) == 1 and round(client.calls[0][0]) == 50
    assert {s.speaker for s in t.segments} == {"0", "1"}  # the model's own labels


def test_server_command_vibevoice_murmur(tmp_path):
    cmd, env = server_command(VibeVoiceClient, port=8123, batch_size=8, speculative="murmur",
                              draft_dir=tmp_path, vllm_bin="vllm", prepare_model=False)
    assert cmd[:3] == ["vllm", "serve", "microsoft/VibeVoice-ASR"] and cmd[-2:] == ["--port", "8123"]
    assert '"method":"ngram"' in cmd[cmd.index("--speculative-config") + 1]
    assert env["MURMUR_MODEL"] == "vibevoice" and env["MURMUR_DRAFT_DIR"] == str(tmp_path)
    assert "_vllm_site" in env["PYTHONPATH"]  # sitecustomize patches the n-gram proposer


def test_server_command_moss_and_greedy(tmp_path):
    cmd, env = server_command(MossClient, port=1, speculative="murmur", draft_dir=tmp_path, vllm_bin="vllm", prepare_model=False)
    assert "MurmurMossProposer" in cmd[cmd.index("--speculative-config") + 1]
    assert "MURMUR_MODEL" not in env
    cmd, env = server_command(MossClient, port=1, speculative=None, vllm_bin="vllm", prepare_model=False)
    assert "--speculative-config" not in cmd and "MURMUR_DRAFT_DIR" not in env
    with pytest.raises(ValueError):
        server_command(MossClient, port=1, speculative="murmur", draft_dir=None, vllm_bin="vllm", prepare_model=False)


def test_evaluate_scores_words_and_speakers():
    meeteval = pytest.importorskip("meeteval")  # noqa: F841
    from murmur.metrics import evaluate
    ref = [{"Speaker ID": "A", "Start time": 0.0, "End time": 2.0, "Content": "hello world"},
           {"Speaker ID": "B", "Start time": 3.0, "End time": 5.0, "Content": "good morning"}]
    hyp = [Segment("x", 0.0, 2.0, "Hello, world!"), Segment("y", 3.0, 5.0, "good evening")]
    sc = evaluate(hyp, ref)
    assert sc["wer"] == pytest.approx(25.0)    # 1 substitution / 4 words, after normalization
    assert sc["cpwer"] == pytest.approx(25.0)  # speakers x->A, y->B
    assert sc["tcpwer"] == pytest.approx(25.0)
    assert math.isnan(sc["der"])               # no RTTM given


def test_parse_stm_keeps_the_first_word(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
    from meeting_data import parse_stm
    f = tmp_path / "m.stm"
    f.write_text(";; header\n"
                 "m 1 A 0.000 2.000 I think so\n"                 # no label field (our STM files)
                 "m 1 B 2.000 3.000 <o,f0,male> Yes, we do\n"     # with a label field
                 "m 1 B 3.000 4.000 okay\n")
    assert [s["Content"] for s in parse_stm(f)] == ["i think so", "yes we do", "okay"]
