"""Persistent records for Worldline evaluations and rollouts."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


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

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_record(run_dir: Path, record: RunRecord) -> None:
    write_json(run_dir / "run.json", asdict(record))
