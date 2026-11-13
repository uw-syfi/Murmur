"""Model clients: request format, output parsing and server flags per model."""

from .base import LocalSegment, ModelClient, Reply
from .moss import MossClient
from .vibevoice import VibeVoiceClient

MODELS: dict[str, type[ModelClient]] = {
    "vibevoice": VibeVoiceClient,
    "vibevoice-asr": VibeVoiceClient,
    "microsoft/VibeVoice-ASR": VibeVoiceClient,
    "moss": MossClient,
    "moss-transcribe-diarize": MossClient,
    "OpenMOSS-Team/MOSS-Transcribe-Diarize": MossClient,
}


def get_model(name: str) -> type[ModelClient]:
    try:
        return MODELS[name if name in MODELS else name.lower()]
    except KeyError:
        raise ValueError(f"Unknown model {name!r}; choose 'vibevoice' or 'moss'.") from None


__all__ = ["LocalSegment", "ModelClient", "Reply", "MossClient", "VibeVoiceClient", "MODELS", "get_model"]
