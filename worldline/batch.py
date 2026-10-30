"""Small, backend-neutral batch orchestration for evaluation runs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
import time
from typing import Sequence
import uuid

from worldline.eval import evaluate_harbor
from worldline.run import write_json


def evaluate_many_harbor(
    task_dirs: Sequence[Path],
    *,
    agent: str,
    model: str | None = None,
    runs_dir: Path = Path("runs"),
    executable: str = "harbor",
    extra_args: Sequence[str] = (),
    dry_run: bool = False,
    max_workers: int = 1,
) -> dict:
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    batch_root = runs_dir.resolve()
    batch_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    def run(task_dir: Path) -> dict:
        try:
            record, run_dir = evaluate_harbor(
                task_dir,
                agent=agent,
                model=model,
                runs_dir=batch_root,
                executable=executable,
                extra_args=extra_args,
                dry_run=dry_run,
            )
            return {
                "task_dir": str(task_dir.resolve()),
                "run_dir": str(run_dir),
                "run": asdict(record),
                "error": None,
            }
        except Exception as error:
            return {
                "task_dir": str(task_dir.resolve()),
                "run_dir": None,
                "run": None,
                "error": f"{type(error).__name__}: {error}",
            }

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run, task): task for task in task_dirs}
        for future in as_completed(futures):
            rows.append(future.result())

    rows.sort(key=lambda row: row["task_dir"])
    statuses: dict[str, int] = {}
    for row in rows:
        status = row["run"]["status"] if row["run"] else "error"
        statuses[status] = statuses.get(status, 0) + 1
    batch_id = f"{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:8]}"
    manifest = {
        "schema_version": "worldline.batch/0.1",
        "batch_id": batch_id,
        "backend": "harbor",
        "agent": agent,
        "model": model,
        "dry_run": dry_run,
        "max_workers": max_workers,
        "tasks": len(task_dirs),
        "statuses": dict(sorted(statuses.items())),
        "runs": rows,
    }
    write_json(batch_root / "batches" / f"{batch_id}.json", manifest)
    return manifest
