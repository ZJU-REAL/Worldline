"""Compatibility helpers for Harbor and Embodied First Exam tasks."""

from __future__ import annotations

import tomllib
from pathlib import Path

from worldline.contracts import Task


def load_task(task_dir: Path) -> Task:
    """Load the stable subset of a Harbor task without rewriting its format."""

    config_path = task_dir / "task.toml"
    instruction_path = task_dir / "instruction.md"
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)

    task_config = config.get("task", {})
    metadata = config.get("metadata", {})
    digest_path = task_dir / "DIGEST"
    version = digest_path.read_text(encoding="utf-8").strip() if digest_path.exists() else "unversioned"

    return Task(
        id=task_config.get("name", task_dir.name),
        instruction=instruction_path.read_text(encoding="utf-8"),
        environment=str(task_dir),
        version=version,
        split=metadata.get("split", "unspecified"),
        seed=metadata.get("seeds", {}).get("default", 0),
        metadata=metadata,
    )
