"""Benchmark one model x dataset x system.

Systems (as in the paper):
  single   whole recording in one request, the model's own speaker labels
  chunked  330 s target chunks, greedy decoding, diarization-linked speakers
  murmur   Murmur: target chunks + draft windows for VibeVoice-ASR, target chunks only for
           MOSS-Transcribe-Diarize (--draft-windows on|off overrides)
  ngram    target chunks + vLLM's built-in n-gram (prompt-lookup) speculative decoding

Modes:
  latency     recordings one at a time, each recording's chunks in parallel; reports
              end-to-end latency per recording, including diarization
  throughput  all chunks of the dataset through one pool of --batch-size requests;
              reports decoding time for the whole dataset

Examples:
  # start a server here (needs the model's vLLM on PATH or --vllm-bin)
  python benchmarks/run.py --model vibevoice --dataset ami_ihm --system murmur --mode latency
  # or use a running one (`murmur serve vibevoice --draft-dir /tmp/murmur_drafts`)
  python benchmarks/run.py --model vibevoice --dataset ami_ihm --system murmur \\
      --url http://localhost:8000 --draft-dir /tmp/murmur_drafts
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import meeting_data as D  # noqa: E402
from murmur import Murmur  # noqa: E402
from murmur.audio import load_audio  # noqa: E402
from murmur.metrics import evaluate  # noqa: E402

log = logging.getLogger("bench")
METRICS = ("wer", "cer", "cpwer", "tcpwer", "der")


def _mean(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(v) / len(v) if v else float("nan")


def diarization_cache(m: Murmur, recs, path: Path) -> list:
    """Diarization timelines for throughput mode, cached on disk (diarization is not timed there)."""
    cache = json.loads(path.read_text()) if path.exists() else {}
    for r in recs:
        if r["id"] not in cache:
            log.info("diarizing %s", r["id"])
            cache[r["id"]] = [list(t) for t in m.diarizer(load_audio(r["wav"]))]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(cache))
    return [cache[r["id"]] for r in recs]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["vibevoice", "moss"])
    ap.add_argument("--dataset", required=True, choices=list(D.DATASETS))
    ap.add_argument("--system", required=True, choices=["single", "chunked", "murmur", "ngram"])
    ap.add_argument("--mode", default="latency", choices=["latency", "throughput"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--chunk-s", type=float, default=330.0)
    ap.add_argument("--draft-windows", choices=["auto", "on", "off"], default="auto",
                    help="with --system murmur; auto = the model's default")
    ap.add_argument("--meetings", nargs="+", default=None)
    ap.add_argument("--url", default=None, help="running server for this --system; default: start one")
    ap.add_argument("--draft-dir", default=None, help="with --url and --system murmur: the server's draft dir")
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--vllm-bin", default=None)
    ap.add_argument("--output", default=None, help="default: results/<model>/<dataset>/<system>_bs<N>_<mode>")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("urllib3", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    out = Path(a.output or HERE.parent / "results" / a.model / a.dataset / f"{a.system}_bs{a.batch_size}_{a.mode}")
    out.mkdir(parents=True, exist_ok=True)
    recs = D.load(a.dataset, a.meetings)
    lang = recs[0]["language"]
    single = a.system == "single"
    draft_windows = (None if a.draft_windows == "auto" else a.draft_windows == "on") if a.system == "murmur" else False
    m = Murmur.load(a.model, url=a.url, draft_dir=a.draft_dir, batch_size=1 if single and a.mode == "latency" else a.batch_size,
                    chunk_s=a.chunk_s, language=lang, gpu=a.gpu, vllm_bin=a.vllm_bin,
                    draft_windows=draft_windows, speculative="ngram" if a.system == "ngram" else None,
                    diarize=not single, single_pass_server=single, keep_failed_output=True)
    try:
        # warm-up, not timed
        clip = load_audio(recs[0]["wav"])[: 16_000 * 60]
        m.transcribe(clip, sr=16_000, single_pass=True)
        if m.diarizer is not None:
            m.diarizer(clip)
        if m.drafter is not None:
            m.drafter(clip.numpy())

        rows = []
        if a.mode == "latency":
            for r in recs:
                t0 = time.perf_counter()
                tr = m.transcribe(r["wav"], single_pass=single)
                rows.append((r, tr, time.perf_counter() - t0))
                log.info("%s: %.1f s, %d chunks", r["id"], rows[-1][2], tr.stats["n_chunks"])
            wall = sum(x[2] for x in rows)
        else:
            tls = None if single else diarization_cache(m, recs, HERE.parent / "results" / "diarization" / f"{a.dataset}.json")
            t0 = time.perf_counter()
            trs = m.transcribe_batch([r["wav"] for r in recs], single_pass=single, timelines=tls)
            wall = time.perf_counter() - t0
            rows = [(r, tr, None) for r, tr in zip(recs, trs)]

        res = []
        for r, tr, lat in rows:
            sc = evaluate(tr, r["ref_segments"], r["ref_rttm"], meeting_id=r["id"], cjk=r["cjk"])
            res.append({"id": r["id"], "duration_s": tr.duration_s, "latency_s": lat,
                        "n_chunks": tr.stats["n_chunks"], "chunks_failed": tr.stats["chunks_failed"],
                        "stages_s": tr.stats["stages_s"], **sc})
        with open(out / "results.jsonl", "w") as f:
            for x in res:
                f.write(json.dumps(x) + "\n")
        decode_s = rows[0][1].stats["stages_s"].get("decoding") if a.mode == "throughput" else None
        summ = {"model": a.model, "dataset": a.dataset, "system": a.system, "mode": a.mode, "batch_size": a.batch_size,
                "chunk_s": a.chunk_s, "n_recordings": len(res), "audio_s": sum(x["duration_s"] for x in res),
                "wall_s": wall, "decoding_s": decode_s,
                "mean_latency_s": _mean([x["latency_s"] for x in res]) if a.mode == "latency" else None,
                "chunks_failed": sum(x["chunks_failed"] for x in res),
                **{k: _mean([x[k] for x in res]) for k in METRICS}}
        (out / "summary.json").write_text(json.dumps(summ, indent=1))
        log.info("%s", json.dumps(summ))
    finally:
        m.close()


if __name__ == "__main__":
    main()
