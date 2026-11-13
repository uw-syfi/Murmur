"""Command line: ``murmur serve`` starts a model's vLLM server; ``murmur transcribe`` transcribes files.

    # in the model's vLLM environment
    murmur serve vibevoice --port 8000 --draft-dir /tmp/murmur_drafts
    # in the client environment
    murmur transcribe meeting.wav --model vibevoice --url http://localhost:8000 --draft-dir /tmp/murmur_drafts
"""

from __future__ import annotations

import argparse
import logging
import os
import sys


def _serve(a) -> int:
    from .models import get_model
    from .server import server_command
    client = get_model(a.model)
    spec = None if a.speculative == "none" else a.speculative
    if spec == "murmur":
        os.makedirs(a.draft_dir, exist_ok=True)
    cmd, env = server_command(client, port=a.port, batch_size=a.batch_size, speculative=spec,
                              draft_dir=a.draft_dir if spec == "murmur" else None, vllm_bin=a.vllm_bin,
                              single_pass=a.single_pass, gpu_memory_utilization=a.gpu_memory_utilization)
    print(" ".join(cmd), file=sys.stderr)
    os.execvpe(cmd[0], cmd, {**os.environ, **env})  # replace this process with vLLM


def _transcribe(a) -> int:
    from .api import Murmur
    with Murmur.load(a.model, url=a.url, draft_dir=a.draft_dir, chunk_s=a.chunk_s, batch_size=a.batch_size,
                     language=a.language, gpu=a.gpu, diarize=not a.no_diarize,
                     draft_windows=None if a.draft_windows == "auto" else a.draft_windows == "on") as m:
        for f in a.files:
            t = m.transcribe(f, single_pass=a.single_pass)
            print(t.to_json(indent=1) if a.json else f"== {f}\n{t}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="murmur", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="start a vLLM server for a model, with Murmur's drafter")
    s.add_argument("model", choices=["vibevoice", "moss"])
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--batch-size", type=int, default=8)
    s.add_argument("--speculative", choices=["murmur", "ngram", "none"], default="murmur")
    s.add_argument("--draft-dir", default="/tmp/murmur_drafts", help="shared with the client")
    s.add_argument("--single-pass", action="store_true", help="MOSS: settings for whole-recording requests")
    s.add_argument("--vllm-bin", default=None)
    s.add_argument("--gpu-memory-utilization", type=float, default=None,
                   help="share of the GPU for vLLM (default 0.70; raise it if the client runs elsewhere)")
    s.set_defaults(func=_serve)

    t = sub.add_parser("transcribe", help="transcribe audio files")
    t.add_argument("files", nargs="+")
    t.add_argument("--model", default="vibevoice", choices=["vibevoice", "moss"])
    t.add_argument("--url", default=None, help="running server; default: start one")
    t.add_argument("--draft-dir", default=None)
    t.add_argument("--draft-windows", choices=["auto", "on", "off"], default="auto")
    t.add_argument("--chunk-s", type=float, default=330.0)
    t.add_argument("--batch-size", type=int, default=8)
    t.add_argument("--language", default="en", choices=["en", "zh"])
    t.add_argument("--gpu", default=None)
    t.add_argument("--no-diarize", action="store_true")
    t.add_argument("--single-pass", action="store_true")
    t.add_argument("--json", action="store_true")
    t.set_defaults(func=_transcribe)

    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
