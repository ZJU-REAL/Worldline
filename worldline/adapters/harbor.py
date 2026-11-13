"""Compatibility helpers for Harbor and Embodied First Exam tasks."""

from __future__ import annotations

import tomllib
import hashlib
import fnmatch
from pathlib import Path
from worldline.storage import file_sha256

from worldline.contracts import Task


def task_fingerprint(task_dir: Path) -> str:
    """Content identity for Harbor task packages without a declared DIGEST."""
    root = task_dir.resolve()
    files = [
        root / name
        for name in ("task.toml", "instruction.md", "README.md", "trajectory.json")
        if (root / name).is_file()
    ]
    for area in ("environment", "tests", "solution", "steps"):
        if (root / area).is_dir():
            files.extend(path for path in (root / area).rglob("*") if path.is_file())
    digest = hashlib.sha256()
    for path in sorted(set(files), key=lambda path: path.relative_to(root).as_posix()):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts or any(
            fnmatch.fnmatch(path.name, pattern) for pattern in ("*.pyc", ".DS_Store", "*.swp", "*.swo", "*~")
        ):
            continue
        if not path.resolve().is_relative_to(root):
            raise ValueError("task package file escapes its root")
        digest.update(f"{relative.as_posix()}\0{file_sha256(path)}\n".encode())
    return "sha256:" + digest.hexdigest()


def load_task(task_dir: Path) -> Task:
    """Load the stable subset of a Harbor task without rewriting its format."""

    config_path = task_dir / "task.toml"
    instruction_path = task_dir / "instruction.md"
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)

    task_config = config.get("task", {})
    metadata = dict(config.get("metadata", {}))
    # Older EFE checkouts declare mode in the directory suffix, not metadata.
    if "mode" not in metadata:
        for mode in ("standard-realtime", "privileged", "standard"):
            if task_dir.name.endswith("-" + mode):
                metadata["mode"] = mode
                break
    digest_path = task_dir / "DIGEST"
    version = digest_path.read_text(encoding="utf-8").strip() if digest_path.exists() else task_fingerprint(task_dir)

    return Task(
        id=task_config.get("name", task_dir.name),
        instruction=instruction_path.read_text(encoding="utf-8"),
        environment=str(task_dir),
        version=version,
        split=metadata.get("split", "unspecified"),
        seed=metadata.get("seeds", {}).get("default", metadata.get("instance_seed", 0)),
        metadata=metadata,
    )
