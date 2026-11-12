"""Lossless, append-only traces used by evaluation and learning pipelines."""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping
from worldline.media import MediaStore


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

    def __init__(self, path: Path, *, sync_every: int = 1, media_store: MediaStore | None = None) -> None:
        if sync_every < 1:
            raise ValueError("sync_every must be positive")
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sync_every = sync_every
        self.media_store = media_store
        self.pending = 0
        self.stream = self.path.open("a", encoding="utf-8")

    def append(self, event: TraceEvent) -> None:
        value = asdict(event)
        if self.media_store is not None:
            value = self.media_store.externalize(value)
        record = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        self.stream.write(record + "\n")
        self.stream.flush()
        self.pending += 1
        if self.pending >= self.sync_every or event.kind.startswith(("run.", "verifier.")):
            os.fsync(self.stream.fileno())
            self.pending = 0

    def close(self):
        if not self.stream.closed:
            self.stream.flush()
            os.fsync(self.stream.fileno())
            self.stream.close()

    def __del__(self):
        if hasattr(self, "stream"):
            self.close()
