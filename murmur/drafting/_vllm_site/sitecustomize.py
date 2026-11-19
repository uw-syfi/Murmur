# Put this directory on the vLLM server's PYTHONPATH (Murmur does this when it
# starts the server). With MURMUR_MODEL set, it turns vLLM's n-gram proposer into
# Murmur's drafter, for vLLM builds without a custom-proposer hook (vLLM 0.14:
# VibeVoice-ASR, Voxtral). MURMUR_LONG_AUDIO=1 keeps long Voxtral audio as one request.
import os

if os.environ.get("MURMUR_MODEL") or os.environ.get("MURMUR_LONG_AUDIO"):
    from murmur.drafting import vllm_proposer as _m

    if os.environ.get("MURMUR_LONG_AUDIO") == "1":
        _m.allow_long_audio()
    if os.environ.get("MURMUR_MODEL"):
        _m.patch_ngram(os.environ["MURMUR_MODEL"])
