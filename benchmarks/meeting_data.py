"""Meeting datasets used in the paper, in a local layout::

    <data>/<name>/test/wav/<id>.wav    16 kHz mono (or any rate; resampled)
    <data>/<name>/test/stm/<id>.stm    reference transcript with speakers and times
    <data>/<name>/test/rttm/<id>.rttm  reference speaker turns (DER)

<data> is ``$MURMUR_DATA`` or ``benchmarks/meeting_datasets``. Build it with the
scripts in ``benchmarks/prepare/`` (see the README).
"""

from __future__ import annotations

import os
from pathlib import Path

from murmur.metrics import normalize

DATA = Path(os.environ.get("MURMUR_DATA", Path(__file__).resolve().parent / "meeting_datasets"))

AMI_TEST = ["EN2002a", "EN2002b", "EN2002c", "EN2002d", "ES2004a", "ES2004b", "ES2004c", "ES2004d",
            "IS1009a", "IS1009b", "IS1009c", "IS1009d", "TS3003a", "TS3003b", "TS3003c", "TS3003d"]

DATASETS = {
    "ami_ihm": {"label": "AMI-IHM", "ids": AMI_TEST, "language": "en"},
    "ami_sdm": {"label": "AMI-SDM", "ids": AMI_TEST, "language": "en"},
    "alimeeting": {"label": "AliMeeting", "ids": None, "language": "zh"},
    "aishell4": {"label": "AISHELL-4", "ids": None, "language": "zh"},
}


def parse_stm(path: Path) -> list[dict]:
    """``<id> <channel> <speaker> <start> <end> [<label>] <text>`` lines (the label is optional)."""
    segs = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(";;"):
            continue
        parts = line.split(None, 5)
        if len(parts) < 6:
            continue
        text = parts[5]
        if text.startswith("<"):  # label field, e.g. <o,f0,male>
            text = text.split(">", 1)[1].strip()
        if not text or text.upper() == "IGNORE_TIME_SEGMENT_IN_SCORING":
            continue
        segs.append({"Speaker ID": parts[2], "Start time": float(parts[3]), "End time": float(parts[4]),
                     "Content": normalize(text)})
    return segs


def parse_rttm(path: Path) -> list[tuple[float, float, str]]:
    out = []
    for line in Path(path).read_text().splitlines():
        p = line.split()
        if len(p) >= 9 and p[0] == "SPEAKER":
            out.append((float(p[3]), float(p[3]) + float(p[4]), p[7]))
    return out


def load(name: str, meetings: list[str] | None = None) -> list[dict]:
    """Recordings of a dataset: dicts with id, wav path, references and language."""
    cfg = DATASETS[name]
    root = DATA / name / "test"
    wav_dir = root / "wav"
    if not wav_dir.is_dir():
        raise FileNotFoundError(f"{wav_dir} not found; prepare {cfg['label']} with benchmarks/prepare/ (README)")
    ids = meetings or cfg["ids"] or sorted(p.stem for p in wav_dir.iterdir() if p.suffix in (".wav", ".flac"))
    out = []
    for mid in ids:
        wav = next((wav_dir / f"{mid}{e}" for e in (".wav", ".flac") if (wav_dir / f"{mid}{e}").exists()), None)
        if wav is None:
            continue
        stm, rttm = root / "stm" / f"{mid}.stm", root / "rttm" / f"{mid}.rttm"
        out.append({"id": mid, "wav": str(wav), "language": cfg["language"], "cjk": cfg["language"] == "zh",
                    "ref_segments": parse_stm(stm) if stm.exists() else [],
                    "ref_rttm": parse_rttm(rttm) if rttm.exists() else []})
    return out
