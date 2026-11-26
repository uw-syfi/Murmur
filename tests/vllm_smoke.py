"""Run explicitly on a CUDA runner with the model's vLLM environment."""

import re

import pytest
import requests

from murmur import Murmur


@pytest.mark.parametrize("model", ["vibevoice", "moss", "voxtral", "voxtral-small"])
def test_vllm_transcription(model, meeting_excerpt, caplog, tmp_path):
    log = tmp_path / "server.log"
    try:
        with Murmur.load(model, diarize=False, server_log=log) as pipeline:
            transcript = pipeline.transcribe(meeting_excerpt)
            if pipeline.draft_windows:
                metrics = requests.get(f"{pipeline.client.url}/metrics", timeout=10)
                metrics.raise_for_status()
                counts = re.findall(r"^vllm:spec_decode_num_draft_tokens_total(?:\{[^}]*\})? (\S+)",
                                    metrics.text, re.MULTILINE)
                assert sum(map(float, counts)) > 0, "vLLM reported no draft tokens"
        assert "short-window ASR failed" not in caplog.text
        assert transcript.stats["n_chunks"] == 1 and transcript.stats["chunks_failed"] == 0
        assert all(term in transcript.text.lower() for term in ("meeting", "project", "remote control"))
        assert all(0 <= s.start <= s.end <= transcript.duration_s for s in transcript.segments)
    except Exception:
        if log.exists():
            print(log.read_text(errors="replace")[-8000:])
        raise
