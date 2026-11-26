"""Opt-in GPU integration test. Set MURMUR_AMI_ES2004A to ES2004a.Mix-Headset.wav."""

import json
import os
import subprocess
import sys

import pytest
import soundfile as sf


@pytest.fixture(scope="session")
def meeting_excerpt(tmp_path_factory):
    source = os.environ.get("MURMUR_AMI_ES2004A")
    if not source:
        pytest.skip("set MURMUR_AMI_ES2004A to run the published MLX checkpoints")
    path = tmp_path_factory.mktemp("ami") / "agenda.wav"
    # The introductions and project brief span multiple 30-second encoder windows.
    with sf.SoundFile(source) as audio:
        audio.seek(80 * audio.samplerate)
        frames = audio.read(round(62.83 * audio.samplerate), dtype="float32")
        sf.write(path, frames, audio.samplerate)
    return path


@pytest.mark.parametrize("precision", ["bf16", "int8", "int4"])
@pytest.mark.parametrize("model", ["moss", "vibevoice"])
def test_published_checkpoint(model, precision, meeting_excerpt, tmp_path):
    # Separate processes release Metal memory between models.
    proc = subprocess.run(
        [sys.executable, "-m", "murmur.cli", "transcribe", str(meeting_excerpt), "--model", model,
         "--backend", "mlx", "--quantization", precision, "--no-diarize", "--json"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "short-window ASR failed" not in proc.stderr
    result = json.loads(proc.stdout)
    (tmp_path / "transcript.json").write_text(proc.stdout)
    assert result["stats"]["n_chunks"] == 1 and result["stats"]["chunks_failed"] == 0
    assert "speculative" not in result["stats"]
    words = " ".join(s["text"] for s in result["segments"]).lower()
    assert all(term in words for term in ("meeting", "project", "remote control")), words
    assert all(0 <= s["start"] <= s["end"] <= result["duration_s"] for s in result["segments"])
