"""Draft windows and the server-side drafter.

Kept import-light on purpose: the vLLM server imports ``murmur.drafting.vllm_proposer``
in an environment that has vLLM and transformers but not the client dependencies.
"""

from pathlib import Path

#: Directory to put on the vLLM server's PYTHONPATH (holds sitecustomize.py).
SITE_DIR = Path(__file__).resolve().parent / "_vllm_site"

VIBEVOICE_SPEC_CONFIG = '{"method":"ngram","num_speculative_tokens":8,"prompt_lookup_max":4,"prompt_lookup_min":2}'
MOSS_SPEC_CONFIG = '{"model":"murmur.drafting.vllm_proposer.MurmurMossProposer","num_speculative_tokens":8}'
NGRAM_SPEC_CONFIG = VIBEVOICE_SPEC_CONFIG  # vLLM's built-in prompt-lookup baseline

__all__ = ["SITE_DIR", "VIBEVOICE_SPEC_CONFIG", "MOSS_SPEC_CONFIG", "NGRAM_SPEC_CONFIG"]
