"""Exercise GGUF routing, audio requests, process ownership and CLI defaults without weights."""

import base64
import io
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf

from murmur import Murmur, cli, server
from murmur.drafting import whisper_drafts
from murmur.models import llama_cpp


def test_vibevoice_audio_request_and_output(monkeypatch):
    response = Mock()
    response.json.return_value = {"choices": [{"message": {"content":
        '[{"Start time":0,"End time":0.8,"Speaker ID":2,"Content":"hello"}]'}, "finish_reason": "stop"}]}
    post = Mock(return_value=response)
    monkeypatch.setattr(llama_cpp.requests, "post", post)
    with Murmur.load("vibevoice-asr", backend="llama.cpp", url="http://test/v1/", served_name="custom",
                     diarize=False) as pipeline:
        assert pipeline.server is None and pipeline.drafter is None
        tr = pipeline.transcribe(np.ones(16000, np.float32) * 0.2, sr=16000, single_pass=True)
    assert tr.text == "hello" and tr.speakers == ["2"] and tr.stats["chunks_failed"] == 0
    assert post.call_args.args == ("http://test/v1/chat/completions",)
    data = post.call_args.kwargs["json"]
    assert data["model"] == "custom" and data["temperature"] == 0
    audio, prompt = data["messages"][0]["content"]
    f = sf.info(io.BytesIO(base64.b64decode(audio["input_audio"]["data"])))
    assert (f.frames, f.samplerate, f.channels) == (24000, 24000, 1)
    assert "1.00 seconds" in prompt["text"]


@pytest.mark.parametrize("model", ["voxtral-mini", "voxtral-small"])
def test_voxtral_uses_language_and_word_times(monkeypatch, model):
    loader = Mock(return_value=lambda audio: [(0.1, "你好")])
    monkeypatch.setattr(whisper_drafts, "WhisperDrafter", loader)
    response = Mock()
    response.json.return_value = {"content": "你好", "stop_type": "eos"}
    props = Mock()
    props.json.return_value = {"media_marker": "<audio_here>"}
    monkeypatch.setattr(llama_cpp.requests, "get", Mock(return_value=props))
    post = Mock(return_value=response)
    monkeypatch.setattr(llama_cpp.requests, "post", post)
    with Murmur.load(model, backend="llama-cpp", url="http://test", language="zh", diarize=False) as pipeline:
        tr = pipeline.transcribe(np.zeros(16000, np.float32), sr=16000, timeline=[(0, 1, "Alice")])
    assert tr.text == "你好" and tr.speakers == ["Alice"]
    assert post.call_args.args == ("http://test/completion",)
    assert post.call_args.kwargs["json"]["prompt"]["prompt_string"] == "[INST]<audio_here>[/INST]lang:zh[TRANSCRIBE]"
    loader.assert_called_once_with("zh", device="cpu", compute_type="int8")


def test_server_command_handles_gguf_pair_and_slot_context(tmp_path):
    client = llama_cpp.LlamaVoxtralClient
    build = server.llama_server_command
    cmd = build(client, port=9000, batch_size=2)
    assert cmd[cmd.index("-hf") + 1] == client.gguf_repo
    assert cmd[cmd.index("--ctx-size") + 1] == str(2 * client.context_per_slot)
    model = tmp_path / "model.gguf"
    projector = tmp_path / "projector.gguf"
    model.touch()
    with pytest.raises(ValueError, match="mmproj"):
        build(client, port=9000, checkpoint=model)
    projector.touch()
    cmd = build(client, port=9000, checkpoint=model, mmproj=projector, llama_bin="/opt/llama-server")
    assert cmd[:5] == ["/opt/llama-server", "-m", str(model), "--mmproj", str(projector)]


def test_cpu_only_places_every_local_model_on_cpu(monkeypatch):
    from murmur import api

    launch = Mock(return_value=Mock(url="http://local"))
    diarizer, whisper = Mock(), Mock()
    monkeypatch.setattr(server, "LlamaServer", launch)
    monkeypatch.setattr(api, "Diarizer", diarizer)
    monkeypatch.setattr(whisper_drafts, "WhisperDrafter", whisper)
    with Murmur.load("voxtral", backend="llama.cpp", cpu_only=True):
        assert launch.call_args.kwargs["cpu_only"] is True
        diarizer.assert_called_once_with(None, device="cpu")
        whisper.assert_called_once_with("en", device="cpu", compute_type="int8")

    cmd = server.llama_server_command(llama_cpp.LlamaVoxtralClient, port=8080, cpu_only=True)
    assert cmd[cmd.index("--device") + 1] == "none"
    assert cmd[cmd.index("--gpu-layers") + 1] == "0"
    assert {"--no-mmproj-offload", "--no-kv-offload", "--no-op-offload"} <= set(cmd)
    assert "--device" not in server.llama_server_command(llama_cpp.LlamaVoxtralClient, port=8080)


def test_failed_setup_stops_owned_server(monkeypatch):
    owned = Mock(url="http://local")
    monkeypatch.setattr(server, "LlamaServer", Mock(return_value=owned))
    monkeypatch.setattr(whisper_drafts, "WhisperDrafter", Mock(side_effect=ImportError("missing")))
    with pytest.raises(ImportError):
        Murmur.load("voxtral", backend="llama.cpp", diarize=False)
    owned.stop.assert_called_once()


@pytest.mark.parametrize(("model", "kwargs"), [
    ("moss", {}), ("vibevoice", {"draft_windows": True}), ("voxtral", {"batch_size": 0}),
    ("voxtral", {"url": "http://test", "checkpoint": "other"}), ("voxtral", {"quantization": "int4"}),
])
def test_invalid_config_rejected_before_launch(monkeypatch, model, kwargs):
    launch = Mock()
    monkeypatch.setattr(server, "LlamaServer", launch)
    with pytest.raises(ValueError):
        Murmur.load(model, backend="llama.cpp", diarize=False, **kwargs)
    launch.assert_not_called()


def test_standalone_serve_cli(monkeypatch):
    def execvp(binary, cmd):
        assert binary == "custom-server"
        assert cmd[cmd.index("--alias") + 1] == "vibevoice"
        assert cmd[cmd.index("--parallel") + 1] == "1"
        assert "--no-mmproj-offload" in cmd
        assert cmd[cmd.index("-hf") + 1] == "test/model:Q8_0"
        raise SystemExit(0)

    monkeypatch.setattr(cli.os, "execvp", execvp)
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["serve", "vibevoice", "--backend", "llama.cpp", "--llama-bin", "custom-server",
                  "--checkpoint", "test/model:Q8_0", "--cpu-only"])
    assert exit_info.value.code == 0


def test_real_meeting_transcription():
    """Exercise an external server or opt in to a fully managed CPU-only run on AMI."""
    import os

    url = os.environ.get("MURMUR_TEST_LLAMA_URL")
    wav = os.environ.get("MURMUR_AMI_ES2004A")
    cpu_model = os.environ.get("MURMUR_TEST_LLAMA_CPU_MODEL")
    if not ((url or cpu_model) and wav):
        pytest.skip("set MURMUR_AMI_ES2004A and MURMUR_TEST_LLAMA_URL or MURMUR_TEST_LLAMA_CPU_MODEL")
    model = cpu_model or os.environ.get("MURMUR_TEST_LLAMA_MODEL", "vibevoice")
    options = {"url": url, "diarize": False}
    if cpu_model:
        options = {"cpu_only": True, "diarize": True, "checkpoint": os.environ.get("MURMUR_TEST_LLAMA_GGUF"),
                   "mmproj": os.environ.get("MURMUR_TEST_LLAMA_MMPROJ")}
    with sf.SoundFile(wav) as f:
        sr = f.samplerate
        f.seek(84 * sr)
        audio = f.read(28 * sr, dtype="float32")
    with Murmur.load(model, backend="llama.cpp", **options) as pipeline:
        if cpu_model:
            assert str(pipeline.diarizer.pipe.device) == "cpu"
            if pipeline.drafter is not None:
                assert pipeline.drafter.pipe.model.model.device == "cpu"
        transcript = pipeline.transcribe(audio, sr=sr)
    assert all(0 <= s.start <= s.end <= len(audio) / sr for s in transcript.segments)
    assert transcript.stats["chunks_failed"] == 0
    assert {"project", "meeting"} <= set(transcript.text.lower().replace(",", "").split())
