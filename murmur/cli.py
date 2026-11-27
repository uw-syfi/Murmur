"""Transcribe audio files, serve the transcription API, or start a vLLM or llama.cpp model server."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys


def _serve(a) -> int:
    from .models import get_model
    from .server import server_command
    client = get_model(a.model)
    if a.backend in ("llama.cpp", "llama-cpp"):
        from .models.llama_cpp import llama_client
        from .server import llama_server_command
        if a.speculative not in (None, "none") or a.vllm_bin or a.gpu_memory_utilization is not None or a.single_pass:
            raise ValueError("Use greedy llama.cpp serving with GGUF weights")
        cmd = llama_server_command(llama_client(client), port=a.port, batch_size=1 if a.batch_size is None else a.batch_size,
                                   checkpoint=a.checkpoint, mmproj=a.mmproj, llama_bin=a.llama_bin)
        os.execvp(cmd[0], cmd)
    if any(v is not None for v in (a.checkpoint, a.mmproj, a.llama_bin)):
        raise ValueError("The serve checkpoint, mmproj and llama-bin flags require --backend llama.cpp")
    spec = None if a.speculative == "none" else (a.speculative or "murmur")
    if spec == "murmur":
        os.makedirs(a.draft_dir, exist_ok=True)
    cmd, env = server_command(client, port=a.port, batch_size=8 if a.batch_size is None else a.batch_size, speculative=spec,
                              draft_dir=a.draft_dir if spec == "murmur" else None, vllm_bin=a.vllm_bin,
                              single_pass=a.single_pass, gpu_memory_utilization=a.gpu_memory_utilization)
    print(" ".join(cmd), file=sys.stderr)
    os.execvpe(cmd[0], cmd, {**os.environ, **env})  # replace this process with vLLM


def _load_pipeline(a):
    from .api import Murmur
    return Murmur.load(a.model, url=a.url, draft_dir=a.draft_dir, chunk_s=a.chunk_s, batch_size=a.batch_size,
                       backend=a.backend, checkpoint=a.checkpoint, quantization=a.quantization,
                       mmproj=a.mmproj, llama_bin=a.llama_bin, served_name=a.served_name,
                       language=a.language, gpu=a.gpu, diarize=not a.no_diarize,
                       draft_windows=None if a.draft_windows == "auto" else a.draft_windows == "on")


def _api(a) -> int:
    from .http_api import serve_api

    def stop(signum, frame):
        # Unwind the model context so a managed vLLM subprocess is stopped as well.
        raise SystemExit(0)

    with _load_pipeline(a) as model:
        previous = signal.signal(signal.SIGTERM, stop)
        try:
            serve_api(model, a.host, a.port, single_pass=a.single_pass)
        finally:
            signal.signal(signal.SIGTERM, previous)
    return 0


def _transcribe(a) -> int:
    with _load_pipeline(a) as m:
        for f in a.files:
            t = m.transcribe(f, single_pass=a.single_pass)
            print(t.to_json(indent=1) if a.json else f"== {f}\n{t}")
    return 0


MODEL_CHOICES = ["vibevoice", "moss", "voxtral", "voxtral-small"]


def _add_pipeline_options(parser):
    parser.add_argument("--model", default="vibevoice", choices=MODEL_CHOICES)
    parser.add_argument("--backend", choices=["vllm", "mlx", "llama.cpp", "llama-cpp"], default="vllm")
    parser.add_argument("--quantization", help="select published MLX weights: bf16 (default), int8 or int4")
    parser.add_argument("--checkpoint", help="override the MLX checkpoint with a local directory or Hugging Face ID, "
                                             "or a llama.cpp GGUF file/Hub repo[:quant]")
    parser.add_argument("--mmproj", help="matching audio projector for a local llama.cpp GGUF")
    parser.add_argument("--llama-bin", help="path to llama-server (default: MURMUR_LLAMA_SERVER or PATH)")
    parser.add_argument("--served-name", help="model alias on an existing server")
    parser.add_argument("--url", default=None, help="running vLLM or llama.cpp server (default: start one)")
    parser.add_argument("--draft-dir", default=None)
    parser.add_argument("--draft-windows", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--chunk-s", type=float, default=330.0)
    parser.add_argument("--batch-size", type=int, help="concurrent chunks (defaults to 8 on vLLM and 1 on MLX and llama.cpp; "
                        "VibeVoice MLX supports batching with drafts)")
    parser.add_argument("--language", default="en", choices=["en", "zh"])
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--no-diarize", action="store_true")
    parser.add_argument("--single-pass", action="store_true")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="murmur", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="start a vLLM or llama.cpp server")
    s.add_argument("model", choices=MODEL_CHOICES)
    s.add_argument("--backend", choices=["vllm", "llama.cpp", "llama-cpp"], default="vllm")
    s.add_argument("--checkpoint", help="GGUF file or Hugging Face GGUF repo[:quant]")
    s.add_argument("--mmproj", help="matching audio projector for a local llama.cpp GGUF")
    s.add_argument("--llama-bin", help="path to llama-server (default: MURMUR_LLAMA_SERVER or PATH)")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--batch-size", type=int, help="parallel slots (vLLM: 8, llama.cpp: 1)")
    s.add_argument("--speculative", choices=["murmur", "ngram", "none"])
    s.add_argument("--draft-dir", default="/tmp/murmur_drafts", help="shared with the client")
    s.add_argument("--single-pass", action="store_true", help="MOSS: settings for whole-recording requests")
    s.add_argument("--vllm-bin", default=None)
    s.add_argument("--gpu-memory-utilization", type=float, default=None,
                   help="share of the GPU for vLLM (default 0.70; raise it if the client runs elsewhere)")
    s.set_defaults(func=_serve)

    t = sub.add_parser("transcribe", help="transcribe audio files")
    t.add_argument("files", nargs="+")
    _add_pipeline_options(t)
    t.add_argument("--json", action="store_true")
    t.set_defaults(func=_transcribe)

    api = sub.add_parser("api", help="serve the Murmur transcription pipeline over HTTP")
    _add_pipeline_options(api)
    api.add_argument("--host", default="127.0.0.1")
    api.add_argument("--port", type=int, default=8080)
    api.set_defaults(func=_api)

    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
