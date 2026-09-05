from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MonitorEvent:
    """A platform-neutral event produced by the DP-GEN monitor."""

    key: str
    event_type: str
    title: str
    message: str
    iteration: int | None = None
    image_paths: tuple[Path, ...] = ()
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        """Stable digest used to suppress identical notifications permanently."""
        content = {
            "event_type": self.event_type,
            "title": self.title,
            "message": self.message,
            "iteration": self.iteration,
            "payload": self.payload,
        }
        encoded = json.dumps(
            content,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
