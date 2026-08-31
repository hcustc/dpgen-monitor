from dataclasses import dataclass, field
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
