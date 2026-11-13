"""Persistent records for Worldline evaluations and rollouts."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping
from worldline.storage import publish_bytes


def make_run_id(task_name: str) -> str:
    slug = task_name.rsplit("/", 1)[-1].replace("_", "-")
    return f"{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{slug}-{uuid.uuid4().hex[:8]}"


@dataclass(frozen=True, slots=True)
class RunRecord:
    run_id: str
    task_id: str
    task_version: str
    backend: str
    agent: str
    model: str | None
    command: tuple[str, ...]
    status: str
    created_at_ns: int = field(default_factory=time.time_ns)
    started_at_ns: int | None = None
    finished_at_ns: int | None = None
    exit_code: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


def write_json(path: Path, value: Any) -> None:
    """Atomically publish JSON so interrupted runs never leave half a record."""

    publish_bytes(path, (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8"))


def write_record(run_dir: Path, record: RunRecord) -> None:
    write_json(run_dir / "run.json", asdict(record))
