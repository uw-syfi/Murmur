# Put this directory on the vLLM server's PYTHONPATH (Murmur does this when it
# starts the server). With MURMUR_MODEL set, it turns vLLM's n-gram proposer
# into Murmur's drafter, for vLLM builds without a custom-proposer hook
# (VibeVoice-ASR's vLLM 0.14).
import os

if os.environ.get("MURMUR_MODEL"):
    from murmur.drafting import vllm_proposer as _m

    _m.patch_ngram(os.environ["MURMUR_MODEL"])
