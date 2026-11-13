"""Batch queue inspection/control, independent of agents and environments."""

from __future__ import annotations

import fcntl
import json
import sqlite3
from pathlib import Path


def validate_batch_id(batch_id: str):
    if not batch_id or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in batch_id
    ):
        raise ValueError("batch_id must contain only letters, digits, hyphens and underscores")


def snapshot_batch(state: Path) -> dict:
    state = state.resolve(strict=True)
    with sqlite3.connect(state.as_uri() + "?mode=ro", uri=True) as db:
        rows = []
        for ordinal, path, status, result in db.execute(
            "SELECT ordinal, path, status, result FROM jobs ORDER BY ordinal"
        ):
            row = json.loads(result) if result else {"task_dir": path, "attempts": []}
            rows.append({**row, "ordinal": ordinal, "status": status})
        control = dict(db.execute("SELECT key, value FROM control"))
    statuses = {}
    for row in rows:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    owned = False
    with state.with_name("owner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            owned = True
    return {
        "batch_id": state.parent.name,
        "owner_active": owned,
        "stop_requested": control.get("stop_requested") == "1",
        "tasks": len(rows),
        "statuses": statuses,
        "runs": rows,
    }


def batch_status(batch_id: str, runs_dir: Path = Path("runs")) -> dict:
    validate_batch_id(batch_id)
    return snapshot_batch(runs_dir / "batches" / batch_id / "state.sqlite")


def request_stop(batch_id: str, runs_dir: Path = Path("runs")) -> dict:
    """Drain active trials; preserve pending trials for an explicit resume."""
    validate_batch_id(batch_id)
    state = (runs_dir / "batches" / batch_id / "state.sqlite").resolve(strict=True)
    with sqlite3.connect(state.as_uri() + "?mode=rw", uri=True, timeout=30) as db:
        db.execute("INSERT OR REPLACE INTO control VALUES ('stop_requested', '1')")
    return snapshot_batch(state)
