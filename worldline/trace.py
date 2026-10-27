"""Lossless, append-only traces used by evaluation and learning pipelines."""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class TraceEvent:
    run_id: str
    kind: str
    payload: Mapping[str, Any]
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    parent_id: str | None = None
    timestamp_ns: int = field(default_factory=time.time_ns)
    schema_version: str = "worldline.trace/0.1"


class TraceWriter:
    """Append events as JSONL and make every append visible to a live runner."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: TraceEvent) -> None:
        record = json.dumps(asdict(event), ensure_ascii=False, separators=(",", ":"))
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(record + "\n")
            stream.flush()
            os.fsync(stream.fileno())
