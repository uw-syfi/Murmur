"""Local HTTP interface for a loaded Murmur pipeline."""

from __future__ import annotations

import json
import logging

from werkzeug.exceptions import HTTPException
from werkzeug.serving import make_server
from werkzeug.wrappers import Request, Response

from .audio import read_audio_file

log = logging.getLogger(__name__)


def _error(message: str, status: int) -> Response:
    return Response(json.dumps({"error": message}), status=status, mimetype="application/json")


def transcription_app(model, *, single_pass=False):
    @Request.application
    def app(request):
        if request.path != "/transcriptions":
            return _error("Unknown endpoint", 404)
        if request.method != "POST":
            response = _error("Use POST to upload a recording", 405)
            response.allow = ["POST"]
            return response
        try:
            file = request.files.get("file")
        except HTTPException as exc:
            return _error(exc.description, exc.code)
        if file is None:
            return _error("Upload a recording in the multipart field 'file'", 400)
        try:
            samples, sr = read_audio_file(file.stream)
        except (RuntimeError, ValueError):
            return _error("Could not decode the uploaded audio", 400)
        if not samples.size:
            return _error("The recording contains no samples", 400)
        try:
            transcript = model.transcribe(samples, sr=sr, single_pass=single_pass)
            return Response(transcript.to_json(), mimetype="application/json")
        except Exception:
            log.exception("transcription failed")
            return _error("Transcription failed", 500)

    return app


def serve_api(model, host: str, port: int, *, single_pass=False) -> None:
    # The single-threaded server keeps MLX inference on the thread that loaded the model.
    with make_server(host, port, transcription_app(model, single_pass=single_pass)) as server:
        log.info("serving transcriptions at http://%s:%d/transcriptions", host, server.server_port)
        server.serve_forever()
