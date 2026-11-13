"""Temporary disk indexes keep deduplication memory independent of dataset size."""

import json
import sqlite3
import tempfile
from pathlib import Path


class DiskIndex:
    def __init__(self):
        self.directory = tempfile.TemporaryDirectory(prefix="worldline-index-")
        # A registry may access its index from serialized HTTP callbacks.
        # Shared callers serialize operations through their owning registry.
        self.db = sqlite3.connect(Path(self.directory.name) / "index.sqlite", check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=OFF")
        self.db.execute("PRAGMA cache_size=-2048")
        self.db.execute("CREATE TABLE entries (key TEXT PRIMARY KEY, value TEXT)")

    def get(self, key, default=None):
        row = self.db.execute("SELECT value FROM entries WHERE key=?", (json.dumps(key, sort_keys=True),)).fetchone()
        return json.loads(row[0]) if row else default

    def add(self, key, value=True):
        cursor = self.db.execute(
            "INSERT OR IGNORE INTO entries VALUES (?, ?)", (json.dumps(key, sort_keys=True), json.dumps(value))
        )
        return bool(cursor.rowcount)

    def set(self, key, value):
        self.db.execute(
            "INSERT OR REPLACE INTO entries VALUES (?, ?)", (json.dumps(key, sort_keys=True), json.dumps(value))
        )

    def count_value(self, value):
        return self.db.execute("SELECT COUNT(*) FROM entries WHERE value=?", (json.dumps(value),)).fetchone()[0]

    def __contains__(self, key):
        return self.get(key) is not None

    def __len__(self):
        return self.db.execute("SELECT COUNT(*) FROM entries").fetchone()[0]

    def close(self):
        self.db.close()
        self.directory.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class ValidationIndex(DiskIndex):
    def __init__(self):
        super().__init__()
        self.db.execute("CREATE TABLE members (kind TEXT, name TEXT, file TEXT, PRIMARY KEY(kind,name,file))")

    def member(self, kind, name, file):
        self.db.execute("INSERT OR IGNORE INTO members VALUES (?,?,?)", (kind, name, file))

    def overlaps(self, kind, limit=1000):
        pairs = self.db.execute(
            "SELECT a.file,b.file,COUNT(*) FROM members a JOIN members b ON a.kind=b.kind AND a.name=b.name AND a.file<b.file WHERE a.kind=? GROUP BY a.file,b.file",
            (kind,),
        ).fetchall()
        output = []
        for left, right, total in pairs:
            names = [
                row[0]
                for row in self.db.execute(
                    "SELECT a.name FROM members a JOIN members b ON a.kind=b.kind AND a.name=b.name WHERE a.kind=? AND a.file=? AND b.file=? ORDER BY a.name LIMIT ?",
                    (kind, left, right, limit),
                )
            ]
            output.append(
                {"left": left, "right": right, kind: names, "shared_count": total, "truncated": total > limit}
            )
        return output


class ErrorLog(list):
    def __init__(self, limit=1000):
        super().__init__()
        self.limit, self.dropped = limit, 0

    def append(self, value):
        if len(self) < self.limit:
            super().append(value)
        else:
            self.dropped += 1
