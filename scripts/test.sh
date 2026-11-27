#!/usr/bin/env bash
# Run Murmur's CI test suites locally, with the same commands as .github/workflows.
#
#   scripts/test.sh [SUITE] [pytest args...]
#
# Suites:
#   cpu         CPU tests, as on every pull request (default)
#   package     build the wheel and smoke-test it in a fresh virtualenv
#   mlx         Voxtral Mini INT4 on Apple Silicon (nightly)      needs MURMUR_AMI_ES2004A
#   mlx-full    all 12 MLX checkpoints (before a release)        needs MURMUR_AMI_ES2004A
#   cuda        VibeVoice on vLLM with speculative decoding     needs MURMUR_AMI_ES2004A, MURMUR_VLLM_*
#   cuda-full   all four vLLM models                             needs MURMUR_AMI_ES2004A, MURMUR_VLLM_*
#   llama-cpu   llama.cpp managed pipeline on CPU (Voxtral)      needs MURMUR_AMI_ES2004A, llama-server
#   benchmark   VibeVoice tcpWER and latency on AMI-IHM ES2004a  needs MURMUR_DATA, MURMUR_VLLM_VIBEVOICE
#
# PYTHON selects the interpreter (default: python); MURMUR_CI_PYTHON is also accepted.
# Extra arguments go to pytest, e.g. `scripts/test.sh cpu -k stitch -x`.
# See docs/ci.md for the environment each hardware suite expects.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-${MURMUR_CI_PYTHON:-python}}
SUITE=${1:-cpu}; [[ $# -gt 0 ]] && shift

need_ami() { test -r "${MURMUR_AMI_ES2004A:?set MURMUR_AMI_ES2004A to the path of ES2004a.Mix-Headset.wav}"; }

case $SUITE in
  cpu)
    "$PY" -m pytest -q --ignore=tests/test_mlx_checkpoints.py "$@" ;;
  package)
    tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
    "$PY" -m pip install -q build twine
    "$PY" -m build -q --outdir "$tmp/dist"
    "$PY" -m twine check --strict "$tmp"/dist/*
    "$PY" -m venv "$tmp/smoke"
    "$tmp/smoke/bin/python" -m pip install -q --upgrade pip
    "$tmp/smoke/bin/python" -m pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    "$tmp/smoke/bin/python" -m pip install -q "$tmp"/dist/*.whl
    "$tmp/smoke/bin/python" -m pip check
    (cd "$tmp" && smoke/bin/murmur --help >/dev/null && smoke/bin/python - <<'PY'
from murmur import Murmur
from murmur.drafting import SITE_DIR

assert (SITE_DIR / "sitecustomize.py").is_file(), SITE_DIR
print("installed wheel OK")
PY
    ) ;;
  mlx)       need_ami; "$PY" -m pytest -q 'tests/test_mlx_checkpoints.py::test_published_checkpoint[voxtral-int4]' "$@" ;;
  mlx-full)  need_ami; "$PY" -m pytest -q tests/test_mlx_checkpoints.py "$@" ;;
  cuda)      need_ami; "$PY" -m pytest -q 'tests/vllm_smoke.py::test_vllm_transcription[vibevoice]' "$@" ;;
  cuda-full) need_ami; "$PY" -m pytest -q tests/vllm_smoke.py "$@" ;;
  llama-cpu)
    need_ami
    MURMUR_TEST_LLAMA_CPU_MODEL=${MURMUR_TEST_LLAMA_CPU_MODEL:-voxtral} "$PY" -m pytest -q tests/test_llama_cpp.py "$@" ;;
  benchmark)
    out=${OUT:-results/ci}
    "$PY" benchmarks/run.py --model vibevoice --dataset ami_ihm --meetings ES2004a \
      --system murmur --mode latency --output "$out"
    OUT=$out "$PY" - <<'PY'
import json, math, os
from pathlib import Path

stats = json.loads((Path(os.environ["OUT"]) / "summary.json").read_text())
for key in ("tcpwer", "chunks_failed", "mean_latency_s"):
    print(f"{key}: {stats[key]}")
assert stats["n_recordings"] == 1 and math.isfinite(stats["tcpwer"]), stats
assert stats["chunks_failed"] == 0, stats
PY
    ;;
  -h|--help|help) sed -n '2,19p' "$0" ;;
  *) echo "unknown suite '$SUITE' (try: scripts/test.sh help)" >&2; exit 2 ;;
esac
