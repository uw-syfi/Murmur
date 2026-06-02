"""Make ``murmur`` and the ``benchmarks/`` scripts importable without an install,
so the suite runs from a bare checkout (``pytest``) and in CI alike."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))
