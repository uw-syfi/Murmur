"""Result types returned by :meth:`murmur.Murmur.transcribe`."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class Segment:
    """One speaker turn: who spoke, when (seconds from the start of the audio), and what."""

    speaker: str
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    """A speaker-attributed, timestamped transcript of one recording."""

    segments: list[Segment]
    duration_s: float = 0.0
    stats: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        """The words only, in time order."""
        return " ".join(s.text for s in self.segments)

    @property
    def speakers(self) -> list[str]:
        return sorted({s.speaker for s in self.segments})

    def to_dict(self) -> dict:
        return {"duration_s": self.duration_s, "segments": [asdict(s) for s in self.segments],
                "stats": self.stats}

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, **kw)

    def __str__(self) -> str:
        return "\n".join(f"[{s.start:7.2f}-{s.end:7.2f}] {s.speaker}: {s.text}" for s in self.segments)
