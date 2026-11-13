"""Evaluation entry points built on existing sandbox backends."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
import hashlib
from dataclasses import asdict, replace
from pathlib import Path
from typing import Mapping, Sequence

from worldline.adapters.harbor import load_task
from worldline.run import RunRecord, write_record, make_run_id
from worldline.run import write_json
from worldline.results import normalize_harbor_trial
from worldline.trace import TraceEvent, TraceWriter


class BackendUnavailableError(RuntimeError):
    pass


def harbor_command(
    task_dir: Path,
    agent: str,
    model: str | None,
    executable: str = "harbor",
    extra_args: Sequence[str] = (),
    jobs_dir: Path | None = None,
    job_name: str | None = None,
) -> tuple[str, ...]:
    command = [executable, "run", "-y", "-p", str(task_dir), "-a", agent]
    if model:
        command.extend(("-m", model))
    if jobs_dir is not None:
        command.extend(("--jobs-dir", str(jobs_dir)))
    if job_name is not None:
        command.extend(("--job-name", job_name))
    command.extend(extra_args)
    return tuple(command)


def _ingest_completed_trials(
    backend_jobs_dir: Path, run_dir: Path, trace: TraceWriter, run_id: str, *, task_snapshot: dict | None = None
) -> int:
    normalized = []
    for result_path in sorted(backend_jobs_dir.rglob("result.json")):
        trial_dir = result_path.parent
        if not (trial_dir / "verifier").is_dir():
            continue
        result = normalize_harbor_trial(trial_dir, task_snapshot=task_snapshot)
        normalized.append(asdict(result))
        trace.append(
            TraceEvent(
                run_id=run_id,
                kind="result.ingested",
                payload={"trial_id": result.trial_id, "outcome": result.outcome, "source": result.source},
            )
        )
    if not normalized:
        return 0
    write_json(run_dir / "results.normalized.json", {"schema_version": "worldline.results/0.1", "results": normalized})
    if len(normalized) == 1:
        write_json(run_dir / "result.normalized.json", normalized[0])
    else:
        for result in normalized:
            key = hashlib.sha256(str(result["source"]).encode()).hexdigest()[:16]
            write_json(run_dir / "trials" / key / "result.normalized.json", result)
    return len(normalized)


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
    task_metadata: Mapping | None = None,
) -> tuple[RunRecord, Path]:
    """Run one Harbor task and preserve a backend-neutral Worldline record."""

    task_dir = task_dir.resolve()
    task = load_task(task_dir)
    if task_metadata:
        split = task_metadata.get("split", task.split)
        if task.split != "unspecified" and split != task.split:
            raise ValueError("catalog cannot override a task's declared split")
        task = replace(task, split=split, metadata={**task.metadata, **task_metadata, "split": split})
    run_id = make_run_id(task.id)
    run_dir = runs_dir.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    trace = TraceWriter(run_dir / "trace.jsonl")
    backend_jobs_dir = run_dir / "harbor"
    command = harbor_command(
        task_dir,
        agent,
        model,
        executable,
        extra_args,
        jobs_dir=backend_jobs_dir,
        job_name="job",
    )
    record = RunRecord(
        run_id=run_id,
        task_id=task.id,
        task_version=task.version,
        backend="harbor",
        agent=agent,
        model=model,
        command=command,
        status="planned" if dry_run else "created",
        metadata={**task.metadata, "task_dir": str(task_dir), "seed": task.seed, "split": task.split},
    )
    write_record(run_dir, record)
    write_json(run_dir / "task.json", asdict(task))
    trace.append(TraceEvent(run_id=run_id, kind="run.created", payload={"command": command}))

    if dry_run:
        trace.append(TraceEvent(run_id=run_id, kind="run.planned", payload={"backend": "harbor"}))
        trace.close()
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
        trace.close()
        raise BackendUnavailableError(f"Harbor executable not found: {executable}")

    started_at = time.time_ns()
    running = replace(record, status="running", started_at_ns=started_at)
    write_record(run_dir, running)
    trace.append(TraceEvent(run_id=run_id, kind="run.started", payload={"started_at_ns": started_at}))

    process_env = os.environ.copy()
    if env:
        process_env.update(env)
    log_path = run_dir / "backend.log"
    try:
        with log_path.open("wb") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=process_env, check=False)
        ingested = _ingest_completed_trials(backend_jobs_dir, run_dir, trace, run_id, task_snapshot=asdict(task))
    except BaseException as error:
        failed = replace(running, status="infra_error", finished_at_ns=time.time_ns(), exit_code=2)
        write_record(run_dir, failed)
        trace.append(TraceEvent(run_id, "run.infra_error", {"error": f"{type(error).__name__}: {error}"}))
        trace.close()
        raise
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
                "normalized_results": ingested,
            },
        )
    )
    trace.close()
    return result, run_dir
