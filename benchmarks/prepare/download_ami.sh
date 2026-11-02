#!/bin/bash
# Original AMI session audio (Mix-Headset = IHM-mix, Array1-01 = SDM) for the
# 16 full-corpus-ASR test meetings, plus the official manual annotations 1.6.2.
DATA="${MURMUR_DATA:-$(cd "$(dirname "$0")/.." && pwd)/meeting_datasets}"
mkdir -p "$DATA/_ami_raw/annotations" && cd "$DATA/_ami_raw"
U=https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus
for m in EN2002a EN2002b EN2002c EN2002d ES2004a ES2004b ES2004c ES2004d IS1009a IS1009b IS1009c IS1009d TS3003a TS3003b TS3003c TS3003d; do
  mkdir -p amicorpus/$m/audio
  for mic in Mix-Headset Array1-01; do
    f=amicorpus/$m/audio/$m.$mic.wav
    [ -s $f ] || curl -sSf -o $f "$U/$m/audio/$m.$mic.wav" || echo "FAILED $f"
  done
  echo "done $m"
done
curl -sSfL -o annotations/ami_public_manual_1.6.2.zip https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip || echo "FAILED annotations"
echo ALL DONE
