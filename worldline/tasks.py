"""Native task manifests, separate from inherited Harbor task packages."""

from __future__ import annotations

import hashlib
import json
import tomllib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from worldline.contracts import Task
from worldline.plugins import get_environment_plugin


@dataclass(frozen=True)
class NativeTask:
    task: Task
    environment_config: Mapping[str, Any]
    timeout_sec: float
    max_actions: int
    max_turns: int
    reset_budget: int
    restore_budget: int = 0


def load_native_task(task_dir: Path) -> NativeTask:
    task_dir = task_dir.resolve()
    manifest = (task_dir / "worldline.toml").read_bytes()
    config = tomllib.loads(manifest.decode("utf-8"))
    if config.get("schema_version") != "worldline.task/0.1":
        raise ValueError("native task requires schema_version = 'worldline.task/0.1'")
    declaration = config["task"]
    plugin = get_environment_plugin(declaration["environment"])
    if declaration["domain"] != plugin.domain:
        raise ValueError("task domain does not match environment plugin")
    if not isinstance(declaration.get("id"), str) or not declaration["id"]:
        raise ValueError("task id must be a non-empty string")
    seed = declaration.get("seed", 0)
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    split = declaration.get("split", "unspecified")
    if split not in {"train", "validation", "test", "development", "unspecified"}:
        raise ValueError(f"unsupported split: {split}")
    budgets = config.get("evaluation", {})
    values = {
        key: budgets.get(key, default)
        for key, default in {
            "timeout_sec": 120,
            "max_actions": 200,
            "max_turns": 100,
            "reset_budget": 0,
            "restore_budget": 0,
        }.items()
    }
    for key, value in values.items():
        minimum = 0 if key in {"reset_budget", "restore_budget"} else 1
        if type(value) is not int or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    instruction = (task_dir / "instruction.md").read_text(encoding="utf-8")
    version = hashlib.sha256(manifest + b"\0" + instruction.encode() + b"\0" + plugin.version.encode()).hexdigest()
    metadata = dict(config.get("metadata", {}))
    resources = metadata.get("resources", {})
    if not isinstance(resources, dict) or type(resources.get("gpu_required", False)) is not bool:
        raise ValueError("resources must be an object with a boolean gpu_required")
    cpus = resources.get("min_cpus", 1)
    if type(cpus) not in {int, float} or not math.isfinite(cpus) or cpus <= 0:
        raise ValueError("min_cpus must be a positive finite number")
    metadata.update(domain=plugin.domain, task_dir=str(task_dir), environment_version=plugin.version)
    # A shared group can bind scene variants, demo rewrites and recovery branches.
    metadata.setdefault("split_group", declaration["id"])
    task = Task(declaration["id"], instruction, plugin.name, version, split, seed, metadata)
    environment_config = config.get("environment", {})
    json.dumps(environment_config, allow_nan=False)
    return NativeTask(task, environment_config, **values)
