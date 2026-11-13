"""Validate training inputs and launch an external trainer with provenance."""

from __future__ import annotations

import hashlib
import subprocess
import sys
import time
import tomllib
import uuid
import os
import signal
from pathlib import Path
from typing import Any

from worldline.datasets import _jsonl, _record_split, validate_datasets
from worldline.run import write_json
from worldline.storage import file_sha256


def train(config_path: Path, *, dry_run: bool = False) -> tuple[dict[str, Any], Path]:
    config_path = config_path.resolve()
    content = config_path.read_bytes()
    document = tomllib.loads(content.decode())
    if document.get("schema_version") != "worldline.training/0.1":
        raise ValueError("unsupported training schema")
    config = dict(document["trainer"])
    if config.get("timeout_sec") is not None and (
        type(config["timeout_sec"]) not in {int, float} or not 0 < config["timeout_sec"] <= 7 * 24 * 3600
    ):
        raise ValueError("trainer timeout must be in (0,604800] seconds")
    for name in ("train_file", "validation_file", "output_dir", "jobs_dir", "working_dir"):
        if config.get(name):
            path = Path(config[name])
            config[name] = str((config_path.parent / path).resolve()) if not path.is_absolute() else str(path.resolve())
    train_file = Path(config["train_file"])
    inputs = [train_file]
    if config.get("validation_file"):
        inputs.append(Path(config["validation_file"]))
    validation = validate_datasets(inputs)
    if not validation["valid"]:
        raise ValueError(f"training dataset validation failed: {validation['errors']}")
    if validation["records"][str(train_file)] == 0:
        raise ValueError("training dataset is empty")
    for _, row in _jsonl(train_file):
        if _record_split(row) in {"test", "evaluation", "held_out", "validation"}:
            raise ValueError("training input contains a held-out or validation record")
        if config["backend"] == "transformers-sft" and not row.get("messages"):
            raise ValueError("transformers-sft expects chat messages")
    if config["backend"] == "transformers-sft":
        if not config.get("model"):
            raise ValueError("transformers-sft requires a model/tokenizer path or identifier")
        command = [sys.executable, "-m", "worldline.learning.transformers_sft", "--config"]
    elif config["backend"] == "command":
        template = config.get("command")
        if not isinstance(template, list) or not template or not all(isinstance(part, str) for part in template):
            raise ValueError("command backend requires a non-empty argument array")
        command = [part.format_map(config) for part in template]
    else:
        raise ValueError(f"unknown trainer backend: {config['backend']}")
    output = Path(config["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise ValueError("model output directory must be empty to avoid mixing training attempts")
    job_id = f"train-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:8]}"
    job_dir = Path(config.get("jobs_dir", output.parent / "training-jobs")) / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    effective = job_dir / "config.json"
    write_json(effective, config)
    if config["backend"] == "transformers-sft":
        command.append(str(effective))
    manifest = {
        "schema_version": "worldline.training_run/0.1",
        "id": job_id,
        "status": "planned" if dry_run else "running",
        "backend": config["backend"],
        "config_sha256": hashlib.sha256(content).hexdigest(),
        "config": config,
        "inputs": [
            {
                "path": str(path),
                "sha256": file_sha256(path),
                "records": validation["records"][str(path)],
            }
            for path in inputs
        ],
        "validation": validation,
        "command": command,
        "created_at_ns": time.time_ns(),
    }
    path = job_dir / "manifest.json"
    write_json(path, manifest)
    if dry_run:
        return manifest, path
    process = None
    try:
        with (job_dir / "trainer.log").open("wb") as log:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                cwd=config.get("working_dir", config_path.parent),
            )
            code = process.wait(timeout=config.get("timeout_sec"))
        manifest["exit_code"] = code
        manifest["status"] = "completed" if code == 0 else "failed"
    except BaseException as error:
        manifest.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=f"{type(error).__name__}: {error}",
        )
        raise
    finally:
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=2)
        manifest["finished_at_ns"] = time.time_ns()
        write_json(path, manifest)
    return manifest, path
