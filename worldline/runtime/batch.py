"""Bounded native rollouts, with one fresh worker per attempt."""

from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from worldline.run import write_json
from worldline.runtime.session import evaluate_native


def evaluate_many_native(task_dirs, *, max_workers=1, runs_dir=Path("runs"), **options):
    if type(max_workers) is not int or max_workers < 1:
        raise ValueError("max_workers must be a positive integer")
    rows, statuses = [], {}

    def run(path):
        try:
            record, directory = evaluate_native(path, runs_dir=runs_dir, **options)
            return {
                "task_dir": str(path.resolve()),
                "run_id": record.run_id,
                "run_dir": str(directory),
                "status": record.status,
            }
        except Exception as error:
            return {"task_dir": str(path.resolve()), "status": "error", "error": f"{type(error).__name__}: {error}"}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(run, Path(path)) for path in task_dirs]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    manifest = {
        "schema_version": "worldline.batch/0.1",
        "backend": "native",
        "tasks": len(rows),
        "max_workers": max_workers,
        "statuses": statuses,
        "runs": sorted(rows, key=lambda r: r["task_dir"]),
    }
    write_json(runs_dir.resolve() / "batches" / f"native-{uuid.uuid4().hex}.json", manifest)
    return manifest
