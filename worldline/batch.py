"""Durable bounded Harbor batches; backend retries remain the upstream runner's."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Sequence
import uuid

from worldline import __version__
from worldline.adapters.harbor import load_task
from worldline.eval import evaluate_harbor, recover_harbor_run
from worldline.run import write_json
from worldline.storage import file_sha256
from worldline.runtime.state import DurableQueue
from worldline.runtime.queue import snapshot_batch, validate_batch_id


def _result_hashes(run_dir):
    return {str(path.relative_to(run_dir)): file_sha256(path) for path in run_dir.rglob("result.normalized.json")}


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
    task_metadata: dict[Path, dict] | None = None,
    batch_id: str | None = None,
    resume: bool = False,
) -> dict:
    if type(max_workers) is not int or max_workers < 1:
        raise ValueError("max_workers must be a positive integer")
    if resume and not batch_id:
        raise ValueError("resume requires an explicit batch_id")
    batch_id = batch_id or f"harbor-{uuid.uuid4().hex}"
    validate_batch_id(batch_id)
    tasks = [path.resolve() for path in task_dirs]
    identities = []
    for path in tasks:
        try:
            identities.append((str(path), load_task(path).version))
        except (OSError, ValueError, KeyError):
            identities.append((str(path), "invalid"))
    metadata = {str(path.resolve()): value for path, value in (task_metadata or {}).items()}
    signature = hashlib.sha256(
        json.dumps(
            {
                "tasks": identities,
                "backend": "harbor",
                "agent": agent,
                "model": model,
                "executable": executable,
                "args": list(extra_args),
                "dry_run": dry_run,
                "task_metadata": metadata,
                "worldline_version": __version__,
            },
            sort_keys=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    batch_root = runs_dir.resolve()
    directory = batch_root / "batches" / batch_id
    queue = DurableQueue(directory)
    try:
        if queue.initialize(tasks, signature, resume):
            saved_rows = list(queue.db.execute("SELECT ordinal, status, result FROM jobs"))
            for ordinal, status, saved in saved_rows:
                row = json.loads(saved) if saved else None
                if row and row.get("run_dir"):
                    for relative, expected in row.get("result_hashes", {}).items():
                        path = Path(row["run_dir"]) / relative
                        if not path.is_file() or file_sha256(path) != expected:
                            raise ValueError(
                                "completed Harbor artifact missing or changed; restore artifacts before resume"
                            )
                    if status in {"running", "orphaned", "created", "cancelled"}:
                        record, terminal = recover_harbor_run(Path(row["run_dir"]))
                        row["run"] = asdict(record)
                        row["result_hashes"] = _result_hashes(Path(row["run_dir"]))
                        queue.save(ordinal, row, record.status if terminal else "pending")
                elif status == "running":
                    queue.db.execute("UPDATE jobs SET status='pending' WHERE ordinal=?", (ordinal,))
            queue.db.commit()
        queue.reset_stop()

        def worker():
            while job := queue.claim():
                ordinal, raw_path, saved = job
                path = Path(raw_path)
                row = (
                    json.loads(saved)
                    if saved
                    else {"task_dir": str(path), "run_dir": None, "run": None, "attempts": [], "error": None}
                )
                status = "error"
                try:

                    def created(record, run_dir):
                        row.update(run_id=record.run_id, run_dir=str(run_dir), run=asdict(record))
                        row["attempts"].append({"run_id": record.run_id, "run_dir": str(run_dir)})
                        queue.save(ordinal, row)

                    record, run_dir = evaluate_harbor(
                        path,
                        agent=agent,
                        model=model,
                        runs_dir=batch_root,
                        executable=executable,
                        extra_args=extra_args,
                        dry_run=dry_run,
                        task_metadata=(task_metadata or {}).get(path),
                        on_created=created,
                    )
                    row.update(
                        run=asdict(record),
                        run_dir=str(run_dir),
                        run_id=record.run_id,
                        error=None,
                        result_hashes=_result_hashes(run_dir),
                    )
                    status = record.status
                except Exception as error:
                    row["error"] = f"{type(error).__name__}: {error}"
                    if row.get("run_dir") and (Path(row["run_dir"]) / "run.json").is_file():
                        row["run"] = json.loads((Path(row["run_dir"]) / "run.json").read_text())
                        status = row["run"]["status"]
                finally:
                    queue.save(ordinal, row, status)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(worker) for _ in range(max_workers)]
            try:
                for future in futures:
                    future.result()
            except KeyboardInterrupt:
                queue.stop()
                raise
        snapshot = snapshot_batch(directory / "state.sqlite")
        manifest = {
            "schema_version": "worldline.batch/0.3",
            "batch_id": batch_id,
            "backend": "harbor",
            "worldline_version": __version__,
            "agent": agent,
            "model": model,
            "dry_run": dry_run,
            "max_workers": max_workers,
            "tasks": len(tasks),
            "statuses": snapshot["statuses"],
            "runs": snapshot["runs"],
            "resume": resume,
            "state": str(directory / "state.sqlite"),
            "retry_owner": "upstream-harbor-runner",
        }
        write_json(directory / "manifest.json", manifest)
        # Keep the 0.1 manifest entry path for existing downstream readers.
        write_json(batch_root / "batches" / f"{batch_id}.json", manifest)
        return manifest
    finally:
        queue.close()
