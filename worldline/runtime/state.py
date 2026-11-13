"""Backend-independent durable job claims and single-owner batch state."""

from __future__ import annotations

import fcntl
import json
import sqlite3
import threading
from pathlib import Path


class DurableQueue:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.lock = (directory / "owner.lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise ValueError("batch is already owned by another runner") from None
        self.db = None
        self.closed = False
        self.mutex = threading.Lock()
        try:
            self.db = sqlite3.connect(directory / "state.sqlite", check_same_thread=False)
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.execute("CREATE TABLE IF NOT EXISTS control (key TEXT PRIMARY KEY, value TEXT)")
            self.db.execute("CREATE TABLE IF NOT EXISTS meta (signature TEXT)")
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS jobs (ordinal INTEGER PRIMARY KEY, path TEXT, status TEXT, result TEXT)"
            )
        except BaseException:
            self.close()
            raise

    def initialize(self, tasks, signature, resume):
        previous = self.db.execute("SELECT signature FROM meta").fetchone()
        if previous:
            if not resume:
                raise ValueError("batch already exists; use resume or a new batch_id")
            if previous[0] != signature:
                raise ValueError("resume configuration/task version differs from the recorded batch")
        else:
            if resume:
                raise ValueError("batch does not exist to resume")
            self.db.execute("INSERT INTO meta VALUES (?)", (signature,))
            self.db.executemany(
                "INSERT INTO jobs VALUES (?, ?, 'pending', NULL)", enumerate(str(path) for path in tasks)
            )
            self.db.commit()
        return bool(previous)

    def reset_stop(self):
        self.db.execute("INSERT OR REPLACE INTO control VALUES ('stop_requested', '0')")
        self.db.commit()

    def stop(self):
        with self.mutex:
            self.db.execute("INSERT OR REPLACE INTO control VALUES ('stop_requested', '1')")
            self.db.commit()

    def claim(self):
        with self.mutex:
            stop = self.db.execute("SELECT value FROM control WHERE key='stop_requested'").fetchone()
            if stop and stop[0] == "1":
                return None
            row = self.db.execute(
                "SELECT ordinal, path, result FROM jobs WHERE status='pending' ORDER BY ordinal LIMIT 1"
            ).fetchone()
            if row:
                self.db.execute("UPDATE jobs SET status='running' WHERE ordinal=?", (row[0],))
                self.db.commit()
            return row

    def save(self, ordinal, row, status="running"):
        with self.mutex:
            self.db.execute(
                "UPDATE jobs SET status=?, result=? WHERE ordinal=?",
                (status, json.dumps(row, allow_nan=False), ordinal),
            )
            self.db.commit()

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.db is not None:
            self.db.close()
        fcntl.flock(self.lock, fcntl.LOCK_UN)
        self.lock.close()
