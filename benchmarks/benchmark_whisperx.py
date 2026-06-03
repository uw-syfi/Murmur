"""
WhisperX benchmark — comparison baseline for Murmur.

Runs whisperx (ASR + forced alignment + pyannote diarization + speaker
assignment) over the same datasets as benchmark.py and reports the same
metrics (WER, CER, cpWER, tcpWER, DER) and timing (inference / latency / RTF).

Checkpoints live in <output_dir>/checkpoints/whisperx/ so re-runs are cheap
and --eval_only re-prints the table without touching the model.

Usage:
  python benchmark_whisperx.py --dataset ami_ihm
  python benchmark_whisperx.py --dataset ami_ihm --meetings EN2002a
  python benchmark_whisperx.py --dataset ami_ihm --eval_only
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from murmur.inference.local_engine import TranscriptSegment

from benchmark import (
    DATASET_CONFIGS,
    compute_all_metrics,
    ckpt_load,
    ckpt_save,
    load_samples,
    normalize,
    _avg,
    _fmt,
    _log_result,
)

log = logging.getLogger(__name__)


# ── WhisperX wrapper ──────────────────────────────────────────────────────────

class WhisperXEngine:
    """Lazy-loaded whisperx pipeline: transcribe → align → diarize → assign."""

    def __init__(
        self,
        model_name: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        batch_size: int = 16,
        language: str | None = "en",
        hf_token: str | None = None,
    ):
        import whisperx  # imported lazily so --eval_only doesn't require it
        from whisperx.diarize import DiarizationPipeline  # not re-exported at top level

        self.whisperx = whisperx
        self.device = device
        self.batch_size = batch_size
        self.language = language
        self.hf_token = hf_token

        log.info(f"Loading whisperx model '{model_name}' on {device} ({compute_type}) ...")
        self.model = whisperx.load_model(
            model_name, device, compute_type=compute_type, language=language
        )

        self._align_cache: dict[str, tuple] = {}

        # Diarization pipeline — only loaded if HF token is available.
        self._diarize = None
        if hf_token:
            log.info("Loading whisperx diarization pipeline ...")
            self._diarize = DiarizationPipeline(token=hf_token, device=device)
        else:
            log.warning("No HF_TOKEN — speaker assignment disabled.")

        log.info("WhisperX engine ready.")

    def _align_model(self, lang: str):
        if lang not in self._align_cache:
            log.info(f"Loading whisperx align model for language '{lang}' ...")
            self._align_cache[lang] = self.whisperx.load_align_model(
                language_code=lang, device=self.device
            )
        return self._align_cache[lang]

    def transcribe(
        self,
        audio: np.ndarray,
    ) -> tuple[list[TranscriptSegment], dict]:
        """Run the full whisperx pipeline. Returns (segments, stage_times)."""
        stage_times: dict[str, float] = {}

        # 1. ASR
        t0 = time.perf_counter()
        result = self.model.transcribe(audio, batch_size=self.batch_size)
        stage_times["transcribe"] = time.perf_counter() - t0

        lang = result.get("language", self.language or "en")

        # 2. Forced alignment for word-level timestamps
        t0 = time.perf_counter()
        align_model, metadata = self._align_model(lang)
        aligned = self.whisperx.align(
            result["segments"], align_model, metadata, audio, self.device,
            return_char_alignments=False,
        )
        stage_times["align"] = time.perf_counter() - t0

        # 3. Diarization + speaker assignment (optional)
        if self._diarize is not None:
            t0 = time.perf_counter()
            diar_segments = self._diarize(audio)
            stage_times["diarize"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            aligned = self.whisperx.assign_word_speakers(diar_segments, aligned)
            stage_times["assign_speakers"] = time.perf_counter() - t0
        else:
            stage_times["diarize"] = 0.0
            stage_times["assign_speakers"] = 0.0

        # 4. Convert to TranscriptSegment list
        segs: list[TranscriptSegment] = []
        for s in aligned.get("segments", []):
            text = str(s.get("text", "")).strip()
            if not text:
                continue
            segs.append(TranscriptSegment(
                speaker_id=str(s.get("speaker", "unknown")),
                start_s=float(s.get("start", 0.0)),
                end_s=float(s.get("end", 0.0)),
                text=text,
            ))
        return segs, stage_times


# ── Per-meeting run ───────────────────────────────────────────────────────────

def run_whisperx(
    samples: list[dict],
    engine: WhisperXEngine | None,
    ckpt_dir: Path,
    eval_only: bool = False,
) -> list[dict]:
    tag = "whisperx"
    results: list[dict] = []

    for s in samples:
        mid = s["meeting_id"]

        cached = ckpt_load(ckpt_dir, tag, mid)
        if cached:
            _log_result("[WX]", mid, cached)
            results.append(cached)
            continue
        if eval_only:
            log.warning(f"[WX] {mid} — no checkpoint, skipping (--eval_only)")
            continue
        if engine is None:
            log.error(f"[WX] {mid} — engine not loaded; cannot run")
            continue

        log.info(f"[WX] {mid} ({s['duration_s']/60:.1f} min) ...")
        try:
            # whisperx wants float32 numpy at 16 kHz mono (we already load at 16k)
            audio = s["audio"]
            if isinstance(audio, torch.Tensor):
                audio = audio.detach().cpu().numpy().astype(np.float32)
            else:
                audio = np.asarray(audio, dtype=np.float32)

            t_wall = time.perf_counter()
            hyp_segs, stage_times = engine.transcribe(audio)
            wall_time = time.perf_counter() - t_wall

            transcribe_s = stage_times.get("transcribe", 0.0)
            align_s = stage_times.get("align", 0.0)
            diar_s = stage_times.get("diarize", 0.0)
            assign_s = stage_times.get("assign_speakers", 0.0)
            inference_s = transcribe_s

            has_speakers = bool(s["ref_segments"]) and assign_s > 0
            metrics = compute_all_metrics(
                hyp_segs, s["ref_text"], s["ref_segments"], s["ref_rttm"],
                has_speaker_info=has_speakers, meeting_id=mid,
            )

            result = {
                "meeting_id":    mid,
                "phase":         "whisperx",
                "duration_s":    s["duration_s"],
                "latency_s":     round(wall_time, 2),
                "inference_s":   round(inference_s, 2),
                "inference_rtf": round(inference_s / s["duration_s"], 4),
                "latency_rtf":   round(wall_time / s["duration_s"], 4),
                "n_segments":    len(hyp_segs),
                "stage_times":   {k: round(v, 3) for k, v in stage_times.items()},
                **metrics,
            }
            ckpt_save(ckpt_dir, tag, result)
            results.append(result)
            _log_result("[WX]", mid, result)

            print(
                f"\n── Timing breakdown  [{mid}  whisperx  "
                f"audio={s['duration_s']/60:.1f}min  segs={len(hyp_segs)}] ──\n"
                f"  transcribe (ASR)    {transcribe_s:8.2f}s\n"
                f"  align (word ts)     {align_s:8.2f}s\n"
                f"  diarize             {diar_s:8.2f}s\n"
                f"  assign speakers     {assign_s:8.2f}s\n"
                f"  ── total            {wall_time:8.2f}s"
            )
        except Exception as e:
            log.error(f"  [WX] {mid} FAILED: {e}", exc_info=True)

    return results


# ── Table (mirrors benchmark.print_table column layout for baseline) ──────────

def print_table(results: list[dict]) -> None:
    by_id = {r["meeting_id"]: r for r in results}
    meetings = sorted(by_id)

    hdr = (
        f"  {'MEETING':<12} {'DUR':>6}  "
        f"{'WER-WX':>9}  {'CER-WX':>9}  {'cpWER-WX':>10}  "
        f"{'tcpWER-WX':>11}  {'DER-WX':>9}  "
        f"{'inf-WX':>8}  {'lat-WX':>8}  {'iRTF-WX':>9}  {'lRTF-WX':>9}"
    )
    sep = "=" * len(hdr)
    print(f"\n{sep}")
    print(f"  WhisperX Benchmark")
    print(sep)
    print(hdr)
    print("-" * len(hdr))

    keys = ["wer", "cer", "cpwer", "tcpwer", "der",
            "inference_s", "latency_s", "inference_rtf", "latency_rtf"]
    accs = {k: [] for k in keys}

    for mid in meetings:
        r = by_id[mid]
        dur = r.get("duration_s", 0)
        row = (
            f"  {mid:<12} {dur/60:>5.1f}m  "
            + "  ".join(_fmt(r.get(k)) for k in ["wer", "cer", "cpwer", "tcpwer", "der"])
            + f"  {_fmt(r.get('inference_s'), '.1f', 's'):>8}"
            + f"  {_fmt(r.get('latency_s'), '.1f', 's'):>8}"
            + f"  {_fmt(r.get('inference_rtf'), '.4f', ''):>9}"
            + f"  {_fmt(r.get('latency_rtf'), '.4f', ''):>9}"
        )
        for k in keys:
            accs[k].append(r.get(k, math.nan))
        print(row)

    print("-" * len(hdr))
    avg_row = (
        f"  {'AVG':<12} {'':>6}  "
        + "  ".join(_fmt(_avg(accs[k])) for k in ["wer", "cer", "cpwer", "tcpwer", "der"])
        + f"  {_fmt(_avg(accs['inference_s']), '.1f', 's'):>8}"
        + f"  {_fmt(_avg(accs['latency_s']), '.1f', 's'):>8}"
        + f"  {_fmt(_avg(accs['inference_rtf']), '.4f', ''):>9}"
        + f"  {_fmt(_avg(accs['latency_rtf']), '.4f', ''):>9}"
    )
    print(avg_row)
    print(f"{sep}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="WhisperX benchmark (Murmur comparison)")

    # Model
    parser.add_argument("--model", default="large-v3",
                        help="WhisperX model name (default: large-v3)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute_type", default="float16",
                        choices=["bfloat16", "float16", "float32",
                                 "int8", "int8_float16", "int8_bfloat16", "int8_float32"],
                        help="faster-whisper compute type for the ASR pass "
                             "(default: float16). Alignment (wav2vec2) and "
                             "diarization (pyannote) run in fp32 regardless.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="WhisperX internal batch size (default: 16)")
    parser.add_argument("--language", default="en",
                        help="Force language code (default: en). Pass empty for auto-detect.")

    # Dataset (same flags as benchmark.py)
    parser.add_argument("--dataset", default="ami_ihm", choices=list(DATASET_CONFIGS))
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--hf_split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--hf_cache_dir", default="./hf_ami_cache")
    parser.add_argument("--meetings", nargs="+", default=None)

    # Diarization
    parser.add_argument("--hf_token", default=None,
                        help="HuggingFace token for pyannote diarization. Falls back to HF_TOKEN env var.")

    # Output
    parser.add_argument("--output_dir", default="./outputs/benchmark")
    parser.add_argument("--eval_only", action="store_true",
                        help="Re-print table from checkpoints without running inference")

    args = parser.parse_args()

    samples = load_samples(
        dataset=args.dataset,
        data_root=args.data_root,
        meetings=args.meetings,
        hf_split=args.hf_split,
        hf_cache_dir=args.hf_cache_dir,
    )
    if not samples:
        log.error("No samples loaded.")
        return

    log.info(f"Loaded {len(samples)} meetings | device: {args.device}")

    out_dir = Path(args.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)

    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or None

    engine: WhisperXEngine | None = None
    if not args.eval_only:
        engine = WhisperXEngine(
            model_name=args.model,
            device=args.device,
            compute_type=args.compute_type,
            batch_size=args.batch_size,
            language=args.language or None,
            hf_token=hf_token,
        )

    results = run_whisperx(samples, engine, ckpt_dir, eval_only=args.eval_only)

    if results:
        print_table(results)
        (out_dir / "results_whisperx.json").write_text(
            json.dumps({"whisperx": results}, indent=2)
        )
        log.info(f"Results saved → {out_dir / 'results_whisperx.json'}")


if __name__ == "__main__":
    main()
