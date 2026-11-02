"""Build AMI-IHM / AMI-SDM test sets from the full AMI session recordings:

  IHM = <meeting>.Mix-Headset.wav  (headset mix, "ihm-mix")
  SDM = <meeting>.Array1-01.wav    (first mic of array 1)

with references from the official manual annotations (1.6.2) via lhotse's AMI
recipe, standard full-corpus-ASR test split (16 meetings). Reference segments
keep their real times, so overlapping speech stays overlapped.

Download first (benchmarks/prepare/download_ami.sh), then:
  python benchmarks/prepare/build_ami_sessions.py
Writes benchmarks/meeting_datasets/ami_{ihm,sdm}/test/{wav,stm,rttm}/
(wav entries are symlinks into _ami_raw/).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from lhotse.recipes.ami import (PARTITIONS, parse_ami_annotations, prepare_audio_single,
                                prepare_supervision_other)

ROOT = Path(os.environ.get("MURMUR_DATA", Path(__file__).resolve().parents[1] / "meeting_datasets"))
RAW = ROOT / "_ami_raw"
MICS = {"ami_ihm": "ihm-mix", "ami_sdm": "sdm"}


def main() -> None:
    logging.basicConfig(level=logging.ERROR)
    ann = parse_ami_annotations(RAW / "annotations", normalize="none")
    for name, mic in MICS.items():
        # prepare_ami's own split loop requires train/dev audio too; only the
        # test sessions are downloaded, so call its building blocks directly.
        audio = prepare_audio_single(list((RAW / "amicorpus").rglob(
            "*Mix-Headset.wav" if mic == "ihm-mix" else "*Array1-01.wav")), mic)
        sups = prepare_supervision_other(audio, ann)
        test_ids = set(PARTITIONS["full-corpus-asr"]["test"])
        recs = audio.filter(lambda r: r.id in test_ids)
        sups = sups.filter(lambda s: s.recording_id in test_ids)
        assert len(recs) == 16, f"{mic}: {len(recs)} test recordings"
        out = ROOT / name / "test"
        for sub in ("wav", "stm", "rttm"):
            (out / sub).mkdir(parents=True, exist_ok=True)
        for rec in recs:
            mid = rec.id
            src = Path(rec.sources[0].source).resolve()
            dst = out / "wav" / f"{mid}.wav"
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(src)
            segs = sorted((s for s in sups if s.recording_id == mid), key=lambda s: s.start)
            with open(out / "stm" / f"{mid}.stm", "w") as f:
                f.write(f";; STM for {mid}: official AMI manual annotations 1.6.2 via lhotse, {mic}, real session times\n")
                for s in segs:
                    if s.text and s.text.strip():
                        f.write(f"{mid} 1 {s.speaker} {s.start:.3f} {s.end:.3f} {s.text.strip()}\n")
            with open(out / "rttm" / f"{mid}.rttm", "w") as f:
                for s in segs:
                    if s.duration > 0:
                        f.write(f"SPEAKER {mid} 1 {s.start:.3f} {s.duration:.3f} <NA> <NA> {s.speaker} <NA> <NA>\n")
            print(f"{name} {mid}: {rec.duration:.0f} s, {len(segs)} segments, "
                  f"{len({s.speaker for s in segs})} speakers")


if __name__ == "__main__":
    main()
