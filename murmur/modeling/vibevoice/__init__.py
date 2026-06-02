"""Vendored VibeVoice ASR model + processor (subset of the upstream
`vibevoice` package).

Adapted from Microsoft's VibeVoice (https://github.com/microsoft/VibeVoice),
used under the MIT License. See the LICENSE and NOTICE files in this directory
for the upstream copyright and a summary of the modifications.

Only the ASR-related modules are re-exported here; the streaming / TTS
modules are still present under `modular/` and `processor/` but are not
imported eagerly to avoid pulling in the diffusion + DPM-solver stack
when only ASR is needed.
"""
from .modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from .modular.configuration_vibevoice import VibeVoiceASRConfig
from .processor.vibevoice_asr_processor import VibeVoiceASRProcessor

__all__ = [
    "VibeVoiceASRForConditionalGeneration",
    "VibeVoiceASRConfig",
    "VibeVoiceASRProcessor",
]
