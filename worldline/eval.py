"""Evaluation entry points built on existing sandbox backends."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

from worldline.adapters.harbor import load_task
from worldline.run import RunRecord, write_record
from worldline.trace import TraceEvent, TraceWriter


class BackendUnavailableError(RuntimeError):
    pass


def make_run_id(task_name: str) -> str:
    slug = task_name.rsplit("/", 1)[-1].replace("_", "-")
    return f"{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{slug}-{uuid.uuid4().hex[:8]}"


def harbor_command(
    task_dir: Path,
    agent: str,
    model: str | None,
    executable: str = "harbor",
    extra_args: Sequence[str] = (),
) -> tuple[str, ...]:
    command = [executable, "run", "-y", "-p", str(task_dir), "-a", agent]
    if model:
        command.extend(("-m", model))
    command.extend(extra_args)
    return tuple(command)


def evaluate_harbor(
    task_dir: Path,
    *,
    agent: str,
    model: str | None = None,
    runs_dir: Path = Path("runs"),
    executable: str = "harbor",
    extra_args: Sequence[str] = (),
    dry_run: bool = False,
    env: Mapping[str, str] | None = None,
) -> tuple[RunRecord, Path]:
    """Run one Harbor task and preserve a backend-neutral Worldline record."""

    task_dir = task_dir.resolve()
    task = load_task(task_dir)
    run_id = make_run_id(task.id)
    run_dir = runs_dir.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    trace = TraceWriter(run_dir / "trace.jsonl")
    command = harbor_command(task_dir, agent, model, executable, extra_args)
    record = RunRecord(
        run_id=run_id,
        task_id=task.id,
        task_version=task.version,
        backend="harbor",
        agent=agent,
        model=model,
        command=command,
        status="planned" if dry_run else "created",
        metadata={"task_dir": str(task_dir), "seed": task.seed, "split": task.split},
    )
    write_record(run_dir, record)
    trace.append(TraceEvent(run_id=run_id, kind="run.created", payload={"command": command}))

    if dry_run:
        trace.append(TraceEvent(run_id=run_id, kind="run.planned", payload={"backend": "harbor"}))
        return record, run_dir

    if shutil.which(executable) is None:
        failed = replace(record, status="infra_error", finished_at_ns=time.time_ns())
        write_record(run_dir, failed)
        trace.append(
            TraceEvent(
                run_id=run_id,
                kind="run.infra_error",
                payload={"error": f"executable not found: {executable}"},
            )
        )
        raise BackendUnavailableError(f"Harbor executable not found: {executable}")

    started_at = time.time_ns()
    running = replace(record, status="running", started_at_ns=started_at)
    write_record(run_dir, running)
    trace.append(TraceEvent(run_id=run_id, kind="run.started", payload={"started_at_ns": started_at}))

    process_env = os.environ.copy()
    if env:
        process_env.update(env)
    log_path = run_dir / "backend.log"
    with log_path.open("wb") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=process_env, check=False)

    finished_at = time.time_ns()
    status = "completed" if completed.returncode == 0 else "failed"
    result = replace(
        running,
        status=status,
        finished_at_ns=finished_at,
        exit_code=completed.returncode,
    )
    write_record(run_dir, result)
    trace.append(
        TraceEvent(
            run_id=run_id,
            kind="run.finished",
            payload={
                "status": status,
                "exit_code": completed.returncode,
                "duration_ns": finished_at - started_at,
                "backend_log": str(log_path),
            },
        )
    )
    return result, run_dir
