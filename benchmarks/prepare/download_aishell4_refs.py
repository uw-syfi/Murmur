"""Convert the AISHELL-4 test set (OpenSLR 111, ``test.tar.gz``) into the local layout
(``<output>/{wav,stm,rttm}``). The training archives are not needed.

Audio is channel 0 of each 8-channel FLAC. The STM comes from the per-speaker TextGrid tiers;
the RTTM is AISHELL-4's own (its speaker labels match the tier names).

  wget https://openslr.trmal.net/resources/111/test.tar.gz
  python benchmarks/prepare/download_aishell4_refs.py --archive test.tar.gz --raw_dir /path/to/extracted
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path

from download_alimeeting_refs import normalize   # shared CJK-safe normalize()


def _extract(archive: Path, raw_dir: Path) -> Path:
    test_root = raw_dir / "test"
    if test_root.is_dir() and any((test_root / "wav").glob("*.flac")):
        print(f"  already extracted at {test_root}")
        return test_root
    print(f"  extracting {archive} -> {raw_dir} ...")
    with tarfile.open(archive) as tf:
        tf.extractall(raw_dir)
    return test_root


def _merge_textgrid(tg_path: Path) -> list[dict]:
    import textgrid

    tg = textgrid.TextGrid.fromFile(str(tg_path))
    rows = []
    for tier in tg.tiers:
        spk = tier.name          # e.g. "006-M" -- matches the shipped RTTM's speaker field
        for iv in tier.intervals:
            text = normalize(str(iv.mark or ""))
            if not text:
                continue
            rows.append({"speaker": spk, "start": float(iv.minTime), "end": float(iv.maxTime), "text": text})
    return sorted(rows, key=lambda r: r["start"])


def write_stm(rows: list[dict], meeting_id: str, out_path: Path) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f";; STM for {meeting_id} (generated from AISHELL-4 TextGrid)\n")
        for r in rows:
            f.write(f"{meeting_id} 1 {r['speaker']} {r['start']:.3f} {r['end']:.3f} {r['text']}\n")


def write_channel0_wav(src_flac: Path, out_path: Path) -> None:
    import soundfile as sf

    data, sr = sf.read(str(src_flac), always_2d=True)   # (frames, channels)
    sf.write(str(out_path), data[:, 0], sr, subtype="PCM_16")


def main() -> None:
    p = argparse.ArgumentParser(description="Convert AISHELL-4 test set into STM+RTTM+wav")
    p.add_argument("--archive", required=True, help="Path to test.tar.gz (already downloaded).")
    p.add_argument("--raw_dir", required=True,
                   help="Where to extract (large -- put it on a big disk/mount, NOT the repo checkout).")
    p.add_argument("--output", default="benchmarks/meeting_datasets/aishell4/test")
    args = p.parse_args()

    raw_dir = Path(args.raw_dir); raw_dir.mkdir(parents=True, exist_ok=True)
    out = Path(args.output)
    wav_dir, stm_dir, rttm_dir = out / "wav", out / "stm", out / "rttm"
    for d in (wav_dir, stm_dir, rttm_dir):
        d.mkdir(parents=True, exist_ok=True)

    test_root = _extract(Path(args.archive), raw_dir)
    tg_dir = test_root / "TextGrid"

    written = 0
    for tg_path in sorted(tg_dir.glob("*.TextGrid")):
        mid = tg_path.stem
        flac_path = test_root / "wav" / f"{mid}.flac"
        rttm_src  = tg_dir / f"{mid}.rttm"
        if not flac_path.is_file():
            print(f"  {mid}: no matching flac -- skipping"); continue
        rows = _merge_textgrid(tg_path)
        if not rows:
            print(f"  {mid}: empty TextGrid -- skipping"); continue
        write_stm(rows, mid, stm_dir / f"{mid}.stm")
        if rttm_src.is_file():
            shutil.copyfile(rttm_src, rttm_dir / f"{mid}.rttm")   # official ground truth, not re-derived
        else:
            log_msg = f"  {mid}: no official .rttm shipped alongside the TextGrid -- DER will be unavailable"
            print(log_msg)
        write_channel0_wav(flac_path, wav_dir / f"{mid}.wav")
        n_spk = len({r["speaker"] for r in rows})
        print(f"  {mid}: {len(rows)} segments, {n_spk} speakers written")
        written += 1

    print(f"\nDone — wrote {written} meetings to {out}")


if __name__ == "__main__":
    main()
