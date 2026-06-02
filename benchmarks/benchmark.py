"""
Murmur benchmark: local inference with VibeVoice ASR.

Phase 0 (baseline) — full meeting audio as one model call (no chunking)
Phase 1 (chunked)  — Murmur Implementation: VAD-chunked, batched locally

Datasets
--------
    meeting_datasets/
      ami_ihm/test/
        wav/   EN2002a.wav  EN2002b.wav  ...
        stm/   EN2002a.stm  EN2002b.stm  ...
        rttm/  EN2002a.rttm EN2002b.rttm ...
      ami_sdm/test/
        wav/   ...
        stm/   ...
        rttm/  ...

Metrics: WER, CER, cpWER, tcpWER, DER  (all via murmur.metrics)
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import concurrent.futures
import sys
import threading
import time
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from murmur.inference.local_engine import LocalBatchEngine, TranscriptSegment
import jiwer

from murmur.metrics.asr_metrics import (
    compute_cpwer, compute_tcpwer, compute_der,
)

log = logging.getLogger(__name__)

_BENCH_DIR = Path(__file__).parent

AMI_TEST_MEETINGS = [
    "EN2002a", "EN2002b", "EN2002c", "EN2002d",
    "ES2004a", "ES2004b", "ES2004c", "ES2004d",
    "IS1009a", "IS1009b", "IS1009c", "IS1009d",
    "TS3003a", "TS3003b", "TS3003c", "TS3003d",
]

DATASET_CONFIGS: dict[str, dict] = {
    "ami_ihm": {
        "label":    "AMI-IHM",
        "subdir":   "ami_ihm",
        "hf_name":  "edinburghcstr/ami",
        "hf_config": "ihm",
        "test_ids": AMI_TEST_MEETINGS,
        "loader":   "ami",
    },
    "ami_sdm": {
        "label":    "AMI-SDM",
        "subdir":   "ami_sdm",
        "hf_name":  "edinburghcstr/ami",
        "hf_config": "sdm",
        "test_ids": AMI_TEST_MEETINGS,
        "loader":   "ami",
    },
    "tedlium3": {
        "label":    "TED-LIUM 3",
        "subdir":   "tedlium3",
        # distil-whisper/tedlium-long-form is the public, long-form variant
        # (each row = one full talk). Avoids LIUM/tedlium gating + per-utterance
        # stitching. Speaker IDs not provided — TED talks are single-speaker so
        # cpWER/tcpWER/DER aren't meaningful anyway; WER/CER is canonical here.
        "hf_name":  "distil-whisper/tedlium-long-form",
        "hf_config": None,
        "test_ids": None,
        "loader":   "tedlium",
    },
    "asr_lb_earnings21": {
        # ASR leaderboard long-form: each row = one full earnings call.
        # No per-utterance timestamps / speaker IDs in the row schema, so the
        # long-form loader is reused; WER/CER is the canonical metric here.
        "label":    "Earnings-21 (ASR leaderboard)",
        "subdir":   "asr_lb_earnings21",
        "hf_name":  "hf-audio/asr-leaderboard-longform",
        "hf_config": "earnings21",
        "test_ids": None,
        "loader":   "tedlium",
    },
    "asr_lb_earnings22": {
        "label":    "Earnings-22 (ASR leaderboard)",
        "subdir":   "asr_lb_earnings22",
        "hf_name":  "hf-audio/asr-leaderboard-longform",
        "hf_config": "earnings22",
        "test_ids": None,
        "loader":   "tedlium",
    }
}

# ── Text helpers ──────────────────────────────────────────────────────────────

# common disfluecy fillers
_FILLERS = re.compile(r'\b(uh+|um+|mm+|hmm+|hm+|mhm)\b')

# Non-lexical annotations: [Human Sounds], [Laughter], <noise>, <breath>, etc.
# Must strip contents too — otherwise "[Human Sounds]" becomes "human sounds"
# after the punctuation pass and gets counted as real hypothesis words.
_NONSPEECH = re.compile(r'\[[^\]]*\]|<[^>]*>')


# normalization steps from whisperx paper
def normalize(text: str) -> str:
    text = text.lower()
    text = _NONSPEECH.sub("", text)
    text = _FILLERS.sub("", text)
    text = re.sub(r"[^\w\s']", "", text)
    return re.sub(r"\s+", " ", text).strip()


def format_ts(t: float) -> str:
    t = max(0.0, t)
    total = int(t)
    frac = t - total
    m, s = divmod(total, 60)
    h, m = divmod(m, 60)
    s_str = f"{s + frac:06.3f}"
    return f"{h}:{m:02d}:{s_str}" if h else f"{m}:{s_str}"


def segs_to_text(segs) -> str:
    if not segs:
        return ""
    if isinstance(segs[0], TranscriptSegment):
        return " ".join(s.text for s in segs)
    return " ".join(str(s.get("Content", "")) for s in segs)


# ── Dataset loaders ───────────────────────────────────────────────────────────

def load_samples(
    dataset: str = "ami_ihm",
    data_root: str | None = None,
    meetings: list[str] | None = None,
    hf_split: str = "test",
    hf_cache_dir: str = "./hf_ami_cache",
) -> list[dict]:
    """Load AMI-IHM, AMI-SDM, or TED-LIUM 3 samples.

    Local layout (if downloaded manually):
        <data_root>/<subdir>/test/wav/   <id>.wav ...
        <data_root>/<subdir>/test/stm/   <id>.stm ...
        <data_root>/<subdir>/test/rttm/  <id>.rttm ...
    """
    cfg = DATASET_CONFIGS.get(dataset)
    if cfg is None:
        raise ValueError(f"Unknown dataset {dataset!r}. Choose from: {list(DATASET_CONFIGS)}")

    # Default data_root = same place the profiling benchmark expects it
    root = Path(data_root or (_BENCH_DIR / "meeting_datasets"))
    wav_dir  = root / cfg["subdir"] / "test" / "wav"
    stm_dir  = root / cfg["subdir"] / "test" / "stm"
    rttm_dir = root / cfg["subdir"] / "test" / "rttm"

    if wav_dir.is_dir():
        log.info(f"Loading {cfg['label']} from {wav_dir}")
        return _load_local(wav_dir, stm_dir, rttm_dir, meetings, cfg["test_ids"])

    log.info(f"{wav_dir} not found — downloading {cfg['label']} from HuggingFace ...")
    if cfg["loader"] == "tedlium":
        return _load_hf_tedlium(cfg["hf_name"], cfg["hf_config"], hf_split, hf_cache_dir, meetings)
    return _load_hf(cfg["hf_name"], cfg["hf_config"], hf_split, hf_cache_dir, meetings, cfg["test_ids"])


def _load_local(
    wav_dir: Path,
    stm_dir: Path,
    rttm_dir: Path,
    meetings: list[str] | None,
    default_ids: list[str] | None = None,
) -> list[dict]:
    """Load from local full-meeting WAV files (same signal as profiling benchmark)."""
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly

    def _load_with_sf(path: str) -> tuple[torch.Tensor, int]:
        """Drop-in replacement for torchaudio.load that uses soundfile (no torchcodec dep)."""
        arr, sr = sf.read(path, dtype="float32", always_2d=True)
        # soundfile returns (frames, channels); torchaudio returns (channels, frames)
        return torch.from_numpy(arr.T).contiguous(), sr

    def _resample_sf(wav: torch.Tensor, src_sr: int, dst_sr: int) -> torch.Tensor:
        if src_sr == dst_sr:
            return wav
        g = gcd(src_sr, dst_sr)
        arr = wav.numpy()
        out = resample_poly(arr, dst_sr // g, src_sr // g, axis=-1).astype("float32")
        return torch.from_numpy(out)

    if meetings:
        target = meetings
    elif default_ids:
        target = default_ids
    else:
        # Auto-discover: take every audio file stem in wav_dir
        target = sorted({
            p.stem for p in wav_dir.iterdir()
            if p.suffix.lower() in (".wav", ".flac", ".mp3")
        })
    samples = []

    for mid in target:
        wav_path = wav_dir / f"{mid}.wav"
        if not wav_path.is_file():
            for ext in (".flac", ".mp3"):
                candidate = wav_dir / f"{mid}{ext}"
                if candidate.is_file():
                    wav_path = candidate
                    break
        if not wav_path.is_file():
            log.warning(f"  {mid}: audio not found in {wav_dir}")
            continue

        wav, sr = _load_with_sf(str(wav_path))
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != 16_000:
            wav = _resample_sf(wav, sr, 16_000)
        wav = wav.squeeze(0)

        stm_path  = stm_dir  / f"{mid}.stm"
        rttm_path = rttm_dir / f"{mid}.rttm"

        ref_segments = _parse_stm(stm_path)  if stm_path.is_file()  else []
        ref_rttm     = _parse_rttm(rttm_path) if rttm_path.is_file() else []
        ref_text     = normalize(" ".join(s["Content"] for s in ref_segments))

        dur = wav.shape[0] / 16_000
        log.info(f"  {mid:<12} {dur/60:.1f} min")
        samples.append({
            "meeting_id":   mid,
            "audio":        wav,
            "duration_s":   dur,
            "ref_text":     ref_text,
            "ref_segments": ref_segments,
            "ref_rttm":     ref_rttm,
        })

    return samples


def _extract_audio(audio) -> tuple[np.ndarray, int, str]:
    """Normalize a `datasets` audio field to (array, sample_rate, path).

    Handles both the legacy dict form ({"array", "sampling_rate", "path"})
    and the newer torchcodec `AudioDecoder` returned by `datasets` >= 3.x.
    """
    # New: torchcodec AudioDecoder
    if hasattr(audio, "get_all_samples"):
        samples = audio.get_all_samples()
        data = samples.data
        arr = data.numpy() if hasattr(data, "numpy") else np.asarray(data)
        arr = arr.astype(np.float32)
        if arr.ndim > 1:                 # (channels, samples) → mono
            arr = arr.mean(axis=0)
        sr = int(samples.sample_rate)
        path = str(getattr(audio, "path", "") or "")
        return arr, sr, path
    # Legacy: dict-like
    arr = np.asarray(audio["array"], dtype=np.float32)
    sr = int(audio["sampling_rate"])
    try:
        path = str(audio["path"] or "")
    except (KeyError, TypeError):
        path = ""
    return arr, sr, path


def _load_hf(
    hf_name: str,
    hf_config: str,
    split: str,
    cache_dir: str,
    meetings: list[str] | None,
    default_ids: list[str] | None = None,
) -> list[dict]:
    """HuggingFace fallback: stitches utterance clips into full meetings."""
    from collections import defaultdict
    from datasets import load_dataset
    import torchaudio

    ds = load_dataset(hf_name, hf_config, split=split, cache_dir=cache_dir,
                      trust_remote_code=True)

    fallback_ids = default_ids or AMI_TEST_MEETINGS
    target = set(meetings or fallback_ids)
    by_meeting: dict[str, list] = defaultdict(list)
    for row in ds:
        mid = row.get("meeting_id", "")
        if mid in target:
            by_meeting[mid].append(row)

    samples = []
    for mid in (meetings or fallback_ids):
        rows = by_meeting.get(mid, [])
        if not rows:
            log.warning(f"  {mid}: no rows in HF dataset")
            continue

        rows.sort(key=lambda r: r.get("begin_time", 0.0))
        _, sr, _ = _extract_audio(rows[0]["audio"])
        total_dur = max(r.get("end_time", 0.0) for r in rows)

        full = np.zeros(int(total_dur * sr), dtype=np.float32)
        for row in rows:
            arr, _, _ = _extract_audio(row["audio"])
            s_i = int(row.get("begin_time", 0.0) * sr)
            e_i = min(s_i + len(arr), len(full))
            full[s_i:e_i] = arr[: e_i - s_i]

        wav = torch.from_numpy(full)
        if sr != 16_000:
            wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, 16_000).squeeze(0)

        ref_segments = [
            {"Speaker ID": str(r.get("speaker_id", "unknown")),
             "Start time": r.get("begin_time", 0.0),
             "End time":   r.get("end_time",   0.0),
             "Content":    normalize(str(r.get("text", "")))}
            for r in rows if r.get("text", "").strip()
        ]
        ref_text = normalize(" ".join(r.get("text", "") for r in rows))
        dur = wav.shape[0] / 16_000
        log.info(f"  {mid:<12} {dur/60:.1f} min  ({len(rows)} utterances — stitched)")
        samples.append({
            "meeting_id":   mid,
            "audio":        wav,
            "duration_s":   dur,
            "ref_text":     ref_text,
            "ref_segments": ref_segments,
            "ref_rttm":     [],
        })

    return samples


def _load_hf_tedlium(
    hf_name: str,
    hf_config: str | None,
    split: str,
    cache_dir: str,
    meetings: list[str] | None,
) -> list[dict]:
    from datasets import load_dataset
    import torchaudio

    kwargs = {"split": split, "cache_dir": cache_dir, "trust_remote_code": True}
    ds = load_dataset(hf_name, hf_config, **kwargs) if hf_config else load_dataset(hf_name, **kwargs)

    wanted = set(meetings) if meetings else None
    samples = []
    seen_ids = []

    for i, row in enumerate(ds):
        arr, sr, audio_path = _extract_audio(row["audio"])

        # Talk identifier: prefer explicit id/file_id, fall back to audio path stem
        tid = str(
            row.get("id")
            or row.get("file_id")
            or Path(audio_path).stem
            or f"talk_{i:03d}"
        )
        seen_ids.append(tid)
        if wanted is not None and tid not in wanted:
            continue

        wav = torch.from_numpy(arr)
        if sr != 16_000:
            wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, 16_000).squeeze(0)

        text = str(row.get("text", "")).strip()
        ref_text = normalize(text)
        dur = wav.shape[0] / 16_000

        log.info(f"  {tid:<40} {dur/60:.1f} min")
        samples.append({
            "meeting_id":   tid,
            "audio":        wav,
            "duration_s":   dur,
            "ref_text":     ref_text,
            "ref_segments": [],   # no per-utterance segmentation in long-form
            "ref_rttm":     [],
        })

    if wanted and not samples:
        log.warning(
            f"None of the requested talks matched. Available IDs in this split: {seen_ids}"
        )

    return samples


def torchaudio_load(path: str) -> tuple[torch.Tensor, int]:
    import torchaudio
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    return wav, sr


def _parse_stm(path: Path) -> list[dict]:
    segs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";;"):
                continue
            parts = line.split(None, 6)
            if len(parts) < 6:
                continue
            text = parts[-1] if len(parts) >= 6 else ""
            if text.startswith("<") or text.upper() == "IGNORE_TIME_SEGMENT_IN_SCORING":
                continue
            segs.append({
                "Speaker ID": parts[2],
                "Start time": float(parts[3]),
                "End time":   float(parts[4]),
                "Content":    normalize(text),
            })
    return segs


def _parse_rttm(path: Path) -> list[tuple[float, float, str]]:
    segs = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 9 and parts[0] == "SPEAKER":
                try:
                    segs.append((float(parts[3]), float(parts[3]) + float(parts[4]), parts[7]))
                except (ValueError, IndexError):
                    pass
    return segs


# ── Pyannote diarization ──────────────────────────────────────────────────────

_pyannote_pipeline = None
_pyannote_lock = threading.Lock()


def build_diarization_timeline(
    audio: torch.Tensor,
    sr: int = 16_000,
    hf_token: str | None = None,
) -> list[tuple[float, float, str]]:
    """Run pyannote speaker-diarization-3.1 on a waveform tensor.
    The pipeline is loaded once and reused across all calls (thread-safe).
    """
    global _pyannote_pipeline

    try:
        from pyannote.audio import Pipeline
    except ImportError:
        log.warning("pyannote.audio not installed - skipping diarization.")
        return []

    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        log.warning("No HF_TOKEN - skipping diarization.")
        return []

    try:
        with _pyannote_lock:
            if _pyannote_pipeline is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                _pyannote_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1", token=token
                ).to(torch.device(device))
        pipeline = _pyannote_pipeline

        waveform = audio.unsqueeze(0) if audio.ndim == 1 else audio
        raw = pipeline({"waveform": waveform, "sample_rate": sr})

        annotation = (
            raw if hasattr(raw, "itertracks") else
            getattr(raw, "diarization", getattr(raw, "annotation", None))
        )
        if annotation is None:
            annotation = next(iter(vars(raw).values()))

        timeline = [
            (turn.start, turn.end, spk)
            for turn, _, spk in annotation.itertracks(yield_label=True)
        ]
        log.info(f"  Diarization: {len(timeline)} speaker turns.")
        return timeline
    except Exception as e:
        log.warning(f"  pyannote diarization failed: {e}")
        return []



# ── Per-layer speech_window config ────────────────────────────────────────────

_LAYER_KEY_ALIASES = {
    "sink":   "speech_sink",   "speech_sink":   "speech_sink",
    "window": "speech_window", "speech_window": "speech_window",
    "delay":  "slide_delay",   "slide_delay":   "slide_delay",
    "rate":   "slide_rate",    "slide_rate":    "slide_rate",
}


def load_layer_ranges(path: str) -> list[dict]:
    """Read a speech_window per-layer JSON config into a list of normalized ranges.

    JSON shape: ``[{"start": int, "end": int|null, "sink": int, "window": int,
    "delay": int, "rate": int}, ...]``. Short keys are mapped to the internal
    long names (``sink → speech_sink`` etc.). Overlap/order validation happens
    inside ``SpeechWindowCache``.
    """
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, list):
        raise ValueError(
            f"--speech_layer_config {path}: expected a JSON array, got {type(raw).__name__}"
        )
    out: list[dict] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(
                f"--speech_layer_config[{i}]: expected an object, got {type(entry).__name__}"
            )
        normalized: dict = {}
        for k, v in entry.items():
            if k in ("start", "end"):
                normalized[k] = v
            elif k in _LAYER_KEY_ALIASES:
                normalized[_LAYER_KEY_ALIASES[k]] = int(v)
            else:
                raise ValueError(
                    f"--speech_layer_config[{i}]: unknown key {k!r}. "
                    f"Allowed: start, end, sink, window, delay, rate"
                )
        out.append(normalized)
    return out


# ── Model loading ─────────────────────────────────────────────────────────────

def build_engine(
    model_path: str,
    device: str = "cuda",
    batch_size: int = 8,
    max_new_tokens: int = 32_768,
    attn_implementation: str = "flash_attention_2",
    cache_policy: str | None = None,
    cache_kwargs: dict | None = None,
) -> "LocalBatchEngine":
    """Load VibeVoice ASR and return a ready LocalBatchEngine."""
    log.info(f"Loading VibeVoice ASR from {model_path} ...")
    if cache_policy:
        log.info(f"KV cache policy: {cache_policy} {cache_kwargs or {}}")
    engine = LocalBatchEngine.from_pretrained(
        model_path,
        device=device,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        attn_implementation=attn_implementation,
        cache_policy=cache_policy,
        cache_kwargs=cache_kwargs,
    )
    log.info("Model loaded.")
    return engine


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _ckpt_path(ckpt_dir: Path, tag: str, meeting_id: str) -> Path:
    p = ckpt_dir / tag / f"{meeting_id}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def ckpt_load(ckpt_dir: Path, tag: str, meeting_id: str) -> dict | None:
    p = _ckpt_path(ckpt_dir, tag, meeting_id)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            log.warning(f"Corrupt checkpoint, ignoring: {p}")
    return None


def ckpt_save(ckpt_dir: Path, tag: str, result: dict) -> None:
    serialisable = {
        k: (None if isinstance(v, float) and math.isnan(v) else v)
        for k, v in result.items()
    }
    _ckpt_path(ckpt_dir, tag, result["meeting_id"]).write_text(
        json.dumps(serialisable, indent=2)
    )


# ── Metric computation ────────────────────────────────────────────────────────

def compute_all_metrics(
    hyp_segs: list[TranscriptSegment],
    ref_text: str,
    ref_segments: list[dict],
    ref_rttm: list[tuple[float, float, str]],
    has_speaker_info: bool,
    meeting_id: str,
) -> dict:
    hyp_text = normalize(segs_to_text(hyp_segs))
    hyp_dicts = [
        {
            "Speaker ID": s.speaker_id,
            "Start time": s.start_s,
            "End time":   s.end_s,
            "Content":    normalize(s.text),
        }
        for s in hyp_segs
    ]
    if ref_text.strip():
        wer = round(jiwer.wer(ref_text, hyp_text or "") * 100, 3)
        cer = round(jiwer.cer(ref_text, hyp_text or "") * 100, 3)
    else:
        wer = cer = math.nan

    if has_speaker_info and ref_segments:
        cpwer  = compute_cpwer(ref_segments, hyp_dicts, meeting_id=meeting_id)
        tcpwer = compute_tcpwer(ref_segments, hyp_dicts, meeting_id=meeting_id)
        der    = compute_der(ref_rttm, hyp_dicts) if ref_rttm else math.nan
    else:
        cpwer = tcpwer = der = math.nan

    return {"wer": wer, "cer": cer, "cpwer": cpwer, "tcpwer": tcpwer, "der": der}


# ── Phase 0: baseline ─────────────────────────────────────────────────────────

def run_baseline(
    samples: list[dict],
    engine: LocalBatchEngine,
    ckpt_dir: Path,
    eval_only: bool = False,
    timelines: dict[str, list] | None = None,
    hf_token: str | None = None,
    no_cache: bool = False,
) -> list[dict]:
    tag = "baseline"
    results = []

    for s in samples:
        mid = s["meeting_id"]

        cached = None if no_cache else ckpt_load(ckpt_dir, tag, mid)
        if cached:
            _log_result("[P0]", mid, cached)
            results.append(cached)
            continue
        if eval_only:
            log.warning(f"[P0] {mid} — no checkpoint, skipping (--eval_only)")
            continue

        log.info(f"[P0] {mid} ({s['duration_s']/60:.1f} min) ...")
        try:
            # 1. Diarization (run first so we can pass the timeline into the
            #    engine for per-(local-speaker) → global-speaker remap)
            t_diar = time.perf_counter()
            if hf_token and timelines is not None and mid not in timelines:
                timelines[mid] = build_diarization_timeline(s["audio"], hf_token=hf_token)
            diar_elapsed = time.perf_counter() - t_diar
            timeline = (timelines or {}).get(mid, [])

            # 2. Single-pass transcription (no VAD, no chunking)
            t0 = time.perf_counter()
            hyp_segs, stats = engine.transcribe_audio(
                s["audio"], src_sr=16_000,
                diarization_timeline=timeline or None,
            )
            transcribe_elapsed = time.perf_counter() - t0
            wall_time = diar_elapsed + transcribe_elapsed

            # 3. Eval
            metrics = compute_all_metrics(
                hyp_segs, s["ref_text"], s["ref_segments"], s["ref_rttm"],
                has_speaker_info=bool(timeline), meeting_id=mid,
            )
            st = stats.stage_times or {}
            toks = int(st.get("tokens_generated", 0))
            max_new_tokens_sum = int(st.get("max_tokens_budget_sum", 0))
            evictions = int(st.get("cache_evictions", 0))
            cache_budget = int(st.get("cache_budget", 0))
            phys_kv = int(st.get("cache_physical_kv", 0))
            logi_kv = int(st.get("cache_logical_kv", 0))
            result = {
                "meeting_id":    mid,
                "phase":         "baseline",
                "duration_s":    s["duration_s"],
                "latency_s":     round(wall_time, 2),
                "inference_s":   round(stats.inference_time_s, 2),
                "inference_rtf": round(stats.inference_time_s / s["duration_s"], 4),
                "latency_rtf":   round(wall_time / s["duration_s"], 4),
                "n_segments":    len(hyp_segs),
                **metrics,
            }
            # KV cache eviction stats (only meaningful when a policy is active):
            # logical = tokens that would exist without eviction (total),
            # physical = tokens actually kept; compression = logical / physical.
            if logi_kv > 0:
                result["cache_logical_kv"]  = logi_kv
                result["cache_physical_kv"] = phys_kv
                result["cache_evictions"]   = evictions
                result["cache_budget"]      = cache_budget
                result["kv_compression"]    = round(logi_kv / phys_kv, 4) if phys_kv > 0 else None
            ckpt_save(ckpt_dir, tag, result)
            results.append(result)
            _log_result("[P0]", mid, result)
            print(
                f"\n── Timing breakdown  [{mid}  baseline  "
                f"audio={s['duration_s']/60:.1f}min] ──\n"
                f"  diarization         {diar_elapsed:8.2f}s\n"
                f"  inference (wall)    {stats.inference_time_s:8.2f}s\n"
                f"  total transcribe    {transcribe_elapsed:8.2f}s"
            )
            if max_new_tokens_sum > 0:
                frac = toks / max_new_tokens_sum
                warn = "  ← decode never EOS'd (hit generation cap)" if frac > 0.95 else ""
                print(f"  decoded tokens      {toks:>8d}  / max_new_tokens {max_new_tokens_sum} ({frac*100:.1f}%){warn}")
            if cache_budget > 0:
                print(f"  KV cache budget     {cache_budget:>8d}   (sink + window, per layer, post-prefill)")
            if evictions > 0:
                print(f"  cache evictions     {evictions:>8d}   (across all layers)")
            if logi_kv > 0:
                kept = phys_kv / logi_kv
                ratio = logi_kv / phys_kv if phys_kv > 0 else float("inf")
                print(
                    f"  KV compression      {phys_kv:>8d}/{logi_kv:<8d} "
                    f"({kept*100:.1f}% kept, {ratio:.2f}× compression)"
                )
            print(f"  ── total            {wall_time:8.2f}s")
        except Exception as e:
            log.error(f"  [P0] {mid} FAILED: {e}", exc_info=True)

    return results


# ── Phase 1: chunked ──────────────────────────────────────────────────────────

def run_chunked(
    samples: list[dict],
    engine: LocalBatchEngine,
    ckpt_dir: Path,
    max_chunk_s: float,
    eval_only: bool = False,
    timelines: dict[str, list] | None = None,
    hf_token: str | None = None,
    no_cache: bool = False,
) -> list[dict]:
    tag = f"chunked_{int(max_chunk_s)}s"
    results = []

    for s in samples:
        mid = s["meeting_id"]

        cached = None if no_cache else ckpt_load(ckpt_dir, tag, mid)
        if cached:
            _log_result(f"[P1/{int(max_chunk_s)}s]", mid, cached)
            results.append(cached)
            continue
        if eval_only:
            log.warning(f"[P1/{tag}] {mid} — no checkpoint, skipping")
            continue

        log.info(f"[P1/{int(max_chunk_s)}s] {mid} ({s['duration_s']/60:.1f} min) ...")
        try:
            # 1. Diarization (run first so we can pass timeline into the engine,
            #    which uses it both to bias chunk boundaries and to remap
            #    VibeVoice's per-chunk speaker IDs to global speakers)
            t_diar = time.perf_counter()
            if hf_token and timelines is not None and mid not in timelines:
                timelines[mid] = build_diarization_timeline(s["audio"], hf_token=hf_token)
            diar_elapsed = time.perf_counter() - t_diar
            timeline = (timelines or {}).get(mid, [])

            # 2. VAD/diarization chunking + batched inference + per-chunk remap
            #    + align/merge (all inside the engine)
            t0 = time.perf_counter()
            hyp_segs, stats = engine.transcribe_tensor(
                s["audio"],
                max_chunk_s=max_chunk_s,
                diarization_timeline=timeline or None,
            )
            transcribe_elapsed = time.perf_counter() - t0

            wall_time = diar_elapsed + transcribe_elapsed

            # 3. Eval
            metrics = compute_all_metrics(
                hyp_segs, s["ref_text"], s["ref_segments"], s["ref_rttm"],
                has_speaker_info=bool(timeline), meeting_id=mid,
            )
            result = {
                "meeting_id":    mid,
                "phase":         "chunked",
                "max_chunk_s":   max_chunk_s,
                "duration_s":    s["duration_s"],
                "n_chunks":      stats.num_chunks,
                "latency_s":     round(wall_time, 2),
                "inference_s":   round(stats.inference_time_s, 2),
                "inference_rtf": round(stats.inference_time_s / s["duration_s"], 4),
                "latency_rtf":   round(wall_time / s["duration_s"], 4),
                "n_segments":    len(hyp_segs),
                **metrics,
            }
            # KV cache eviction stats, summed across batches × layers (see baseline).
            _cst = stats.stage_times or {}
            _logi = int(_cst.get("cache_logical_kv", 0))
            _phys = int(_cst.get("cache_physical_kv", 0))
            if _logi > 0:
                result["cache_logical_kv"]  = _logi
                result["cache_physical_kv"] = _phys
                result["cache_evictions"]   = int(_cst.get("cache_evictions", 0))
                result["cache_budget"]      = int(_cst.get("cache_budget", 0))
                result["kv_compression"]    = round(_logi / _phys, 4) if _phys > 0 else None
            ckpt_save(ckpt_dir, tag, result)
            results.append(result)
            _log_result(f"[P1/{int(max_chunk_s)}s]", mid, result)
            _print_timing_breakdown(
                mid, max_chunk_s, s["duration_s"], stats,
                transcribe_elapsed, diar_elapsed,
            )
        except Exception as e:
            log.error(f"  [P1/{tag}] {mid} FAILED: {e}", exc_info=True)

    return results


# ── Logging helper ────────────────────────────────────────────────────────────

def _print_timing_breakdown(
    meeting_id: str,
    max_chunk_s: float,
    duration_s: float,
    stats,
    transcribe_s: float,
    diar_s: float,
) -> None:
    """Print per-stage wall-clock breakdown for a single meeting."""
    st = stats.stage_times or {}
    vad = st.get("vad_chunking", 0.0)
    pre = st.get("batch_preprocess", 0.0)
    gen = st.get("model_generate", 0.0)
    dec = st.get("decode_postprocess", 0.0)
    align = st.get("align_merge", 0.0)
    nb = int(st.get("num_batches", 0))
    toks = int(st.get("tokens_generated", 0))
    max_new_tokens_sum = int(st.get("max_tokens_budget_sum", 0))
    evictions = int(st.get("cache_evictions", 0))
    cache_budget = int(st.get("cache_budget", 0))
    phys_kv = int(st.get("cache_physical_kv", 0))
    logi_kv = int(st.get("cache_logical_kv", 0))
    total = transcribe_s + diar_s

    def _pct(x: float) -> str:
        return f"{(x / total * 100) if total > 0 else 0:5.1f}%"

    print(
        f"\n── Timing breakdown  [{meeting_id}  chunk={int(max_chunk_s)}s  "
        f"audio={duration_s/60:.1f}min  chunks={stats.num_chunks}  batches={nb}] ──"
    )
    print(f"  diarization              {diar_s:8.2f}s  {_pct(diar_s)}")
    print(f"  vad/diar chunking        {vad:8.2f}s  {_pct(vad)}")
    print(f"  batch_preprocess         {pre:8.2f}s  {_pct(pre)}   (sum across batches)")
    print(f"  model_generate           {gen:8.2f}s  {_pct(gen)}   (sum across batches)")
    print(f"  decode_postprocess       {dec:8.2f}s  {_pct(dec)}   (sum across batches)")
    print(f"  align+remap+merge        {align:8.2f}s  {_pct(align)}")
    print(f"  inference loop (wall)    {stats.inference_time_s:8.2f}s  "
          f"{_pct(stats.inference_time_s)}   (batch loop wall time)")
    if max_new_tokens_sum > 0:
        frac = toks / max_new_tokens_sum if max_new_tokens_sum > 0 else 0.0
        warn = "  ← decode never EOS'd (hit generation cap)" if frac > 0.95 else ""
        print(f"  decoded tokens           {toks:>8d}  / max_new_tokens {max_new_tokens_sum} ({frac*100:.1f}%){warn}")
    if cache_budget > 0:
        print(f"  KV cache budget          {cache_budget:>8d}   (sink + window, per layer, post-prefill)")
    if evictions > 0:
        print(f"  cache evictions (sum)    {evictions:>8d}   (across all batches × layers)")
    if logi_kv > 0:
        kept = phys_kv / logi_kv
        ratio = logi_kv / phys_kv if phys_kv > 0 else float("inf")
        print(
            f"  KV compression           {phys_kv:>8d}/{logi_kv:<8d} "
            f"({kept*100:.1f}% kept, {ratio:.2f}× compression, summed across batches × layers)"
        )
    print(f"  ── total                 {total:8.2f}s  100.0%")


def _log_result(prefix: str, meeting_id: str, r: dict) -> None:
    def _f(key):
        v = r.get(key, math.nan)
        return f"{v:.3f}%" if isinstance(v, float) and not math.isnan(v) else "—"
    log.info(
        f"{prefix} {meeting_id}  "
        f"WER={_f('wer')}  CER={_f('cer')}  cpWER={_f('cpwer')}  "
        f"tcpWER={_f('tcpwer')}  DER={_f('der')}  "
        f"inf={r.get('inference_s', 0):.1f}s  lat={r.get('latency_s', 0):.1f}s  "
        f"iRTF={r.get('inference_rtf', math.nan):.4f}  lRTF={r.get('latency_rtf', math.nan):.4f}"
        + (f"  chunks={r['n_chunks']}" if "n_chunks" in r else "")
    )


# ── Results table ─────────────────────────────────────────────────────────────

def _fmt(v, fmt=".3f", unit="%") -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:{fmt}}{unit}"


def _avg(lst: list) -> float:
    lst = [x for x in lst if isinstance(x, float) and not math.isnan(x)]
    return sum(lst) / len(lst) if lst else math.nan


def print_table(baseline: list[dict], chunked_by_size: dict[float, list[dict]]) -> None:
    bl = {r["meeting_id"]: r for r in baseline}
    sizes = sorted(chunked_by_size.keys())
    ch_maps = {sz: {r["meeting_id"]: r for r in chunked_by_size[sz]} for sz in sizes}
    meetings = sorted(set(list(bl) + [m for cm in ch_maps.values() for m in cm]))

    base_hdr = (
        f"  {'MEETING':<12} {'DUR':>6}  "
        f"{'WER-P0':>9}  {'CER-P0':>9}  {'cpWER-P0':>10}  "
        f"{'tcpWER-P0':>11}  {'DER-P0':>9}  "
        f"{'inf-P0':>8}  {'lat-P0':>8}  {'iRTF-P0':>9}  {'lRTF-P0':>9}"
    )
    chunk_hdrs = "".join(
        f"  {'WER-'+f'{int(sz)}s':>10}  {'CER-'+f'{int(sz)}s':>10}  "
        f"{'cpWER-'+f'{int(sz)}s':>12}  {'tcpWER-'+f'{int(sz)}s':>13}  "
        f"{'DER-'+f'{int(sz)}s':>11}  "
        f"{'inf-'+f'{int(sz)}s':>9}  {'lat-'+f'{int(sz)}s':>9}  "
        f"{'iRTF-'+f'{int(sz)}s':>11}  {'lRTF-'+f'{int(sz)}s':>11}  {'CHK':>5}"
        for sz in sizes
    )
    hdr = base_hdr + chunk_hdrs
    sep = "=" * len(hdr)
    print(f"\n{sep}")
    print(f"  Murmur Benchmark  |  Baseline vs Chunked ({', '.join(f'{int(s)}s' for s in sizes)})")
    print(sep)
    print(hdr)
    print("-" * len(hdr))

    keys = ["wer", "cer", "cpwer", "tcpwer", "der", "inference_s", "latency_s", "inference_rtf", "latency_rtf"]
    accs0 = {k: [] for k in keys}
    accs1 = {sz: {k: [] for k in keys} for sz in sizes}

    for mid in meetings:
        r0 = bl.get(mid)
        dur = (r0 or next((ch_maps[sz].get(mid) for sz in sizes if mid in ch_maps[sz]), {})).get("duration_s", 0)
        row = (
            f"  {mid:<12} {dur/60:>5.1f}m  "
            + "  ".join(_fmt(r0.get(k) if r0 else math.nan) for k in ["wer","cer","cpwer","tcpwer","der"])
            + f"  {_fmt(r0.get('inference_s') if r0 else math.nan, '.1f', 's'):>8}"
            + f"  {_fmt(r0.get('latency_s') if r0 else math.nan, '.1f', 's'):>8}"
            + f"  {_fmt(r0.get('inference_rtf') if r0 else math.nan, '.4f', ''):>9}"
            + f"  {_fmt(r0.get('latency_rtf') if r0 else math.nan, '.4f', ''):>9}"
        )
        if r0:
            for k in keys:
                accs0[k].append(r0.get(k, math.nan))

        for sz in sizes:
            r1 = ch_maps[sz].get(mid)
            row += (
                "  " + "  ".join(_fmt(r1.get(k) if r1 else math.nan) for k in ["wer","cer","cpwer","tcpwer","der"])
                + f"  {_fmt(r1.get('inference_s') if r1 else math.nan, '.1f', 's'):>9}"
                + f"  {_fmt(r1.get('latency_s') if r1 else math.nan, '.1f', 's'):>9}"
                + f"  {_fmt(r1.get('inference_rtf') if r1 else math.nan, '.4f', ''):>11}"
                + f"  {_fmt(r1.get('latency_rtf') if r1 else math.nan, '.4f', ''):>11}"
                + f"  {r1['n_chunks'] if r1 and 'n_chunks' in r1 else '—':>5}"
            )
            if r1:
                for k in keys:
                    accs1[sz][k].append(r1.get(k, math.nan))

        print(row)

    print("-" * len(hdr))
    avg_row = (
        f"  {'AVG':<12} {'':>6}  "
        + "  ".join(_fmt(_avg(accs0[k])) for k in ["wer","cer","cpwer","tcpwer","der"])
        + f"  {_fmt(_avg(accs0['inference_s']), '.1f', 's'):>8}"
        + f"  {_fmt(_avg(accs0['latency_s']), '.1f', 's'):>8}"
        + f"  {_fmt(_avg(accs0['inference_rtf']), '.4f', ''):>9}"
        + f"  {_fmt(_avg(accs0['latency_rtf']), '.4f', ''):>9}"
    )
    for sz in sizes:
        avg_row += (
            "  " + "  ".join(_fmt(_avg(accs1[sz][k])) for k in ["wer","cer","cpwer","tcpwer","der"])
            + f"  {_fmt(_avg(accs1[sz]['inference_s']), '.1f', 's'):>9}"
            + f"  {_fmt(_avg(accs1[sz]['latency_s']), '.1f', 's'):>9}"
            + f"  {_fmt(_avg(accs1[sz]['inference_rtf']), '.4f', ''):>11}"
            + f"  {_fmt(_avg(accs1[sz]['latency_rtf']), '.4f', ''):>11}"
            + f"  {'':>5}"
        )
    print(avg_row)

    # ── Final KV-cache compression rate (only when an eviction policy ran) ──────
    def _compression_summary(label: str, rows: list[dict]) -> str | None:
        logi = sum(int(r.get("cache_logical_kv", 0)) for r in rows)
        phys = sum(int(r.get("cache_physical_kv", 0)) for r in rows)
        if logi <= 0:
            return None
        kept = phys / logi
        ratio = logi / phys if phys > 0 else float("inf")
        return (
            f"  {label:<14} total KV {logi:>10d}  →  kept {phys:>10d}   "
            f"({kept*100:.1f}% kept, {ratio:.2f}× compression)"
        )

    comp_lines = []
    bl_line = _compression_summary("baseline", list(bl.values()))
    if bl_line:
        comp_lines.append(bl_line)
    for sz in sizes:
        line = _compression_summary(f"chunked {int(sz)}s", list(ch_maps[sz].values()))
        if line:
            comp_lines.append(line)
    if comp_lines:
        print("-" * len(hdr))
        print("  KV-cache compression (total tokens summed across meetings × layers):")
        for line in comp_lines:
            print(line)

    print(f"{sep}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Murmur benchmark: VibeVoice ASR on AMI-IHM / AMI-SDM"
    )

    # Model
    parser.add_argument("--model_path", default="microsoft/VibeVoice-ASR",
                        help="VibeVoice model — HF hub ID or local path (default: microsoft/VibeVoice-ASR)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="VAD chunks per GPU batch (default: 8)")
    parser.add_argument("--max_new_tokens", type=int, default=32_768)
    parser.add_argument("--attn_implementation", default="sdpa",
                        choices=["flash_attention_2", "sdpa", "eager"],
                        help="Attention backend. flash_attention_2 needs flash-attn installed.")
    parser.add_argument("--cache_policy", default=None,
                        choices=["streaming", "speech_window", "speech_stream"],
                        help="KV cache eviction policy (default: HF DynamicCache, no eviction). "
                             "'streaming' = StreamingLLM-style sink+window over decoded tokens; "
                             "prefill is always preserved. "
                             "'speech_window' = sink+window+slide over the SPEECH-token region "
                             "of the prefill; decoded tokens are preserved. "
                             "'speech_stream' = speech_window AND StreamingLLM together — evicts "
                             "both the prefill speech region and the decoded tokens. "
                             "All support batch_size>1 on the model.generate engine.")
    parser.add_argument("--cache_sink_size", type=int, default=32,
                        help="StreamingCache: number of attention-sink tokens kept from the start "
                             "of the decoded region (default: 32).")
    parser.add_argument("--cache_window_size", type=int, default=1024,
                        help="StreamingCache: sliding-window size over recent decoded tokens (default: 1024).")
    parser.add_argument("--speech_sink", type=int, default=32,
                        help="speech_window: number of speech tokens pinned at the start of the "
                             "speech region (default: 32).")
    parser.add_argument("--speech_window", type=int, default=512,
                        help="speech_window: live sliding-window size at the tail of the speech "
                             "region (default: 512).")
    parser.add_argument("--slide_delay", type=int, default=32,
                        help="speech_window: decode steps before eviction begins (default: 32).")
    parser.add_argument("--slide_rate", type=int, default=8,
                        help="speech_window: speech tokens evicted per decode step after the delay "
                             "(default: 8).")
    parser.add_argument("--speech_evict_block", type=int, default=1,
                        help="speech_window / speech_stream: evict speech tokens in blocks of "
                             "this many instead of every step. With a low --slide_rate the "
                             "speech eviction otherwise fires a full-cache gather on EVERY "
                             "decode step until its cap; a block of N cuts that firing "
                             "frequency by ~N×. Default 1 = original per-step behavior.")
    parser.add_argument("--output_sink", type=int, default=8,
                        help="speech_stream: number of earliest decoded tokens pinned as "
                             "attention sinks (StreamingLLM part, default: 8).")
    parser.add_argument("--output_window", type=int, default=512,
                        help="speech_stream: sliding-window size over the most recent decoded "
                             "tokens (StreamingLLM part, default: 512).")
    parser.add_argument("--output_evict_block", type=int, default=1,
                        help="streaming / speech_stream: evict decoded (output) tokens in "
                             "blocks of this many instead of 1-per-step. The decoded-token "
                             "eviction otherwise fires a full-cache rebuild on EVERY decode "
                             "step once past the window; a block of N cuts that firing "
                             "frequency by N× (the window overshoots by up to N-1 tokens "
                             "between firings). Default 1 = original per-step behavior.")
    parser.add_argument("--speech_layer_config", default=None,
                        help="Path to a JSON file with per-layer-range overrides for the "
                             "speech_window cache policy. Each entry: "
                             "{start, end, sink, window, delay, rate} where end is exclusive "
                             "(null = to last layer). Ranges must not overlap; any key omitted "
                             "inherits the matching --speech_*/--slide_* scalar default. "
                             "Layers not covered by any range also fall back to the scalars.")

    # Dataset
    parser.add_argument("--dataset", default="ami_ihm", choices=list(DATASET_CONFIGS),
                        help="Which dataset to evaluate on (default: ami_ihm)")
    parser.add_argument("--data_root", default=None,
                        help="Root of meeting_datasets/ directory.  "
                             "Expects <data_root>/<dataset>/test/{wav,stm,rttm}/  "
                             "Defaults to benchmarks/meeting_datasets/.  "
                             "Falls back to HuggingFace download if not found.")
    parser.add_argument("--hf_split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--hf_cache_dir", default="./hf_ami_cache")
    parser.add_argument("--meetings", nargs="+", default=None,
                        help="Subset of meeting IDs to run (default: all 16 AMI test meetings)")

    # Benchmark mode
    parser.add_argument("--mode", default="chunked", choices=["baseline", "chunked", "both"])
    parser.add_argument("--max_chunk_s", type=float, nargs="+", default=[300.0],
                        help="Max chunk size(s) in seconds (default: 300).  Pass multiple to compare, e.g. 15 30 60")

    # Diarization
    parser.add_argument("--hf_token", default=None,
                        help="HuggingFace token for pyannote diarization. Falls back to HF_TOKEN env var.")

    # Output
    parser.add_argument("--output_dir", default="./outputs/benchmark")
    parser.add_argument("--eval_only", action="store_true",
                        help="Re-print table from checkpoints without running inference")
    parser.add_argument("--no_cache", action="store_true",
                        help="Ignore existing per-meeting checkpoints and always rerun inference. "
                             "Fresh results still overwrite the checkpoint files.")

    args = parser.parse_args()

    # ── Load dataset ──────────────────────────────────────────────────────────
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

    # ── Load model ────────────────────────────────────────────────────────────
    cache_kwargs = None
    if args.cache_policy == "streaming":
        cache_kwargs = {
            "sink_size": args.cache_sink_size,
            "window_size": args.cache_window_size,
            "output_evict_block": args.output_evict_block,
        }
    elif args.cache_policy == "speech_window":
        cache_kwargs = {
            "speech_sink": args.speech_sink,
            "speech_window": args.speech_window,
            "slide_delay": args.slide_delay,
            "slide_rate": args.slide_rate,
            "speech_evict_block": args.speech_evict_block,
        }
        if args.speech_layer_config:
            ranges = load_layer_ranges(args.speech_layer_config)
            cache_kwargs["layer_ranges"] = ranges
            log.info(
                f"speech_window per-layer overrides: {len(ranges)} ranges "
                f"from {args.speech_layer_config}"
            )
    elif args.cache_policy == "speech_stream":
        cache_kwargs = {
            "speech_sink": args.speech_sink,
            "speech_window": args.speech_window,
            "slide_delay": args.slide_delay,
            "slide_rate": args.slide_rate,
            "output_sink": args.output_sink,
            "output_window": args.output_window,
            "output_evict_block": args.output_evict_block,
            "speech_evict_block": args.speech_evict_block,
        }

    if not args.eval_only:
        engine = build_engine(
            model_path=args.model_path,
            device=args.device,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            attn_implementation=args.attn_implementation,
            cache_policy=args.cache_policy,
            cache_kwargs=cache_kwargs,
        )
    else:
        engine = None  # type: ignore

    out_dir = Path(args.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)

    # timelines cache: populated lazily per meeting inside each run function
    timelines: dict[str, list] = {}
    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or None

    baseline_results: list[dict] = []
    chunked_by_size: dict[float, list[dict]] = {}

    if args.mode in ("baseline", "both"):
        log.info("=== Phase 0: Baseline (full audio, single forward pass) ===")
        baseline_results = run_baseline(
            samples, engine, ckpt_dir,
            eval_only=args.eval_only,
            timelines=timelines,
            hf_token=hf_token,
            no_cache=args.no_cache,
        )

    if args.mode in ("chunked", "both"):
        if args.eval_only:
            # Discover every chunk size that has checkpoints on disk so
            # `--eval_only` re-prints results for all sizes previously tested,
            # not just the ones passed via --max_chunk_s.
            discovered: set[float] = set()
            if ckpt_dir.is_dir():
                for p in ckpt_dir.iterdir():
                    if not p.is_dir():
                        continue
                    m = re.fullmatch(r"chunked_(\d+)s", p.name)
                    if m:
                        discovered.add(float(m.group(1)))
            chunk_sizes = sorted(discovered | set(args.max_chunk_s)) if discovered else sorted(set(args.max_chunk_s))
            log.info(f"[--eval_only] chunk sizes from checkpoints: {[int(s) for s in chunk_sizes]}")
        else:
            chunk_sizes = sorted(set(args.max_chunk_s))

        for chunk_s in chunk_sizes:
            log.info(f"=== Phase 1: Chunked (max={chunk_s}s, batch={args.batch_size}) ===")
            chunked_by_size[chunk_s] = run_chunked(
                samples, engine, ckpt_dir,
                max_chunk_s=chunk_s,
                eval_only=args.eval_only,
                timelines=timelines,
                hf_token=hf_token,
                no_cache=args.no_cache,
            )

    if baseline_results or chunked_by_size:
        print_table(baseline_results, chunked_by_size)

        # Aggregate KV-cache compression (total tokens summed across meetings ×
        # layers), saved alongside the per-meeting rows so the final rate is in
        # results.json too — not just printed. Empty when no eviction policy ran.
        def _agg_compression(rows: list[dict]) -> dict | None:
            logi = sum(int(r.get("cache_logical_kv", 0)) for r in rows)
            phys = sum(int(r.get("cache_physical_kv", 0)) for r in rows)
            if logi <= 0:
                return None
            return {
                "total_logical_kv":  logi,
                "total_physical_kv": phys,
                "kept_fraction":     round(phys / logi, 4),
                "compression":       round(logi / phys, 4) if phys > 0 else None,
            }

        kv_summary: dict = {}
        bl_comp = _agg_compression(baseline_results)
        if bl_comp:
            kv_summary["baseline"] = bl_comp
        for sz, rows in chunked_by_size.items():
            comp = _agg_compression(rows)
            if comp:
                kv_summary[f"chunked_{int(sz)}s"] = comp

        payload = {"baseline": baseline_results, "chunked": chunked_by_size}
        if kv_summary:
            payload["kv_compression"] = kv_summary
        (out_dir / "results.json").write_text(json.dumps(payload, indent=2))
        log.info(f"Results saved → {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
