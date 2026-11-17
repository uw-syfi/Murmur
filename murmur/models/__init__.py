"""Model clients: request format, output parsing and server flags per model."""

from .base import DiarizingModel, LocalSegment, ModelClient, Reply, TranscriptModel
from .moss import MossClient
from .vibevoice import VibeVoiceClient
from .voxtral import VoxtralClient, VoxtralSmallClient

MODELS: dict[str, type[ModelClient]] = {
    "vibevoice": VibeVoiceClient,
    "vibevoice-asr": VibeVoiceClient,
    "microsoft/VibeVoice-ASR": VibeVoiceClient,
    "moss": MossClient,
    "moss-transcribe-diarize": MossClient,
    "OpenMOSS-Team/MOSS-Transcribe-Diarize": MossClient,
    "voxtral": VoxtralClient,
    "voxtral-mini": VoxtralClient,
    "mistralai/Voxtral-Mini-3B-2507": VoxtralClient,
    "voxtral-small": VoxtralSmallClient,
    "mistralai/Voxtral-Small-24B-2507": VoxtralSmallClient,
}


def get_model(name: str) -> type[ModelClient]:
    try:
        return MODELS[name if name in MODELS else name.lower()]
    except KeyError:
        raise ValueError(f"Unknown model {name!r}; choose 'vibevoice', 'moss', 'voxtral' or 'voxtral-small'.") from None


__all__ = ["DiarizingModel", "TranscriptModel", "LocalSegment", "ModelClient", "Reply", "MossClient",
           "VibeVoiceClient", "VoxtralClient", "VoxtralSmallClient", "MODELS", "get_model"]
