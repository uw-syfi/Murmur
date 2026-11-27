"""Exercise the API CLI and multipart uploads without loading model weights."""

import io
from contextlib import nullcontext
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf
from werkzeug.test import Client

from murmur import cli, http_api
from murmur.api import Murmur
from murmur.types import Segment, Transcript


@pytest.mark.parametrize("backend", ["vllm", "mlx"])
def test_api_loads_selected_backend(backend, monkeypatch):
    model = MagicMock()
    load = MagicMock(return_value=nullcontext(model))
    serve = MagicMock()
    monkeypatch.setattr(Murmur, "load", load)
    monkeypatch.setattr(http_api, "serve_api", serve)

    assert cli.main(["api", "--backend", backend, "--model", "moss", "--port", "8123",
                     "--no-diarize", "--single-pass"]) == 0
    load.assert_called_once()
    assert load.call_args.args == ("moss",)
    assert load.call_args.kwargs["backend"] == backend
    assert load.call_args.kwargs["diarize"] is False
    serve.assert_called_once_with(model, "127.0.0.1", 8123, single_pass=True)


def test_uploads_return_transcript_json_from_loaded_pipeline():
    model = MagicMock()
    transcript = Transcript([Segment("A", 0.0, 1.0, "hello")], 1.0)
    model.transcribe.return_value = transcript
    client = Client(http_api.transcription_app(model, single_pass=True))

    for _ in range(2):
        wav = io.BytesIO()
        sf.write(wav, np.full((8000, 2), [0.25, 0.75]), 8000, format="WAV")
        wav.seek(0)
        response = client.post("/transcriptions", data={"file": (wav, "meeting.wav")})
        assert response.status_code == 200
        assert response.json == transcript.to_dict()
    assert model.transcribe.call_count == 2
    for call in model.transcribe.call_args_list:
        np.testing.assert_array_equal(call.args[0], np.full(8000, 0.5))
        assert call.kwargs == {"sr": 8000, "single_pass": True}


def test_invalid_upload_returns_json_without_transcribing():
    model = MagicMock()
    client = Client(http_api.transcription_app(model))
    for data in ({}, {"file": (io.BytesIO(b"bad audio"), "meeting.wav")}):
        response = client.post("/transcriptions", data=data)
        assert response.status_code == 400 and response.json["error"]
    assert client.get("/transcriptions").status_code == 405
    assert client.post("/unknown").status_code == 404
    model.transcribe.assert_not_called()
