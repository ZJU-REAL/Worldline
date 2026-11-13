"""Central task leases for multiple runners; expired work becomes orphaned."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import threading
import time
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class Coordinator:
    def __init__(self, path):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS leases (job TEXT PRIMARY KEY, signature TEXT, owner TEXT, expires REAL, status TEXT, result TEXT)"
        )
        self.lock = threading.Lock()

    def call(self, operation, payload):
        with self.lock:
            now = time.time()
            self.db.execute("UPDATE leases SET status='orphaned' WHERE status='running' AND expires < ?", (now,))
            self.db.commit()
            job = payload["job"]
            row = self.db.execute(
                "SELECT signature,owner,expires,status,result FROM leases WHERE job=?", (job,)
            ).fetchone()
            if operation == "claim":
                if row and row[0] != payload["signature"]:
                    raise ValueError("distributed job signature mismatch")
                if row:
                    return {"granted": False, "status": row[3], "result": json.loads(row[4]) if row[4] else None}
                seconds = payload.get("lease_sec", 60)
                if type(seconds) not in {int, float} or not 1 <= seconds <= 3600:
                    raise ValueError("lease seconds must be in [1,3600]")
                self.db.execute(
                    "INSERT INTO leases VALUES (?, ?, ?, ?, 'running', NULL)",
                    (job, payload["signature"], payload["owner"], now + seconds),
                )
                self.db.commit()
                return {"granted": True, "status": "running"}
            if operation == "status":
                self.db.commit()
                return {
                    "status": row[3] if row else "missing",
                    "result": json.loads(row[4]) if row and row[4] else None,
                }
            if row is None or row[1] != payload["owner"]:
                raise ValueError("distributed lease owner mismatch")
            if operation == "heartbeat":
                if row[3] != "running":
                    raise ValueError("distributed lease expired; stop executing this job")
                self.db.execute("UPDATE leases SET expires=? WHERE job=?", (now + payload.get("lease_sec", 60), job))
            elif operation == "complete":
                if row[3] not in {"running", "orphaned"}:
                    raise ValueError("distributed job is already finalized")
                self.db.execute(
                    "UPDATE leases SET status='completed', result=? WHERE job=?",
                    (json.dumps(payload["result"], allow_nan=False), job),
                )
            else:
                raise ValueError("unknown coordinator operation")
            self.db.commit()
            return {"status": "completed" if operation == "complete" else "running"}


class RemoteLease:
    def __init__(self, endpoint, job, signature, *, api_key_env="WORLDLINE_COORDINATOR_KEY", lease_sec=60):
        from urllib.parse import urlparse

        parsed = urlparse(endpoint)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("coordinator endpoint must be a plain HTTP(S) URL")
        self.endpoint, self.job, self.signature = endpoint.rstrip("/"), job, signature
        self.owner, self.seconds, self.key_env = uuid.uuid4().hex, lease_sec, api_key_env
        self.stop = threading.Event()
        self.error = None
        self.thread = None

    def call(self, operation, **values):
        body = {"job": self.job, "signature": self.signature, "owner": self.owner, "lease_sec": self.seconds, **values}
        headers = {"Content-Type": "application/json"}
        if os.environ.get(self.key_env):
            headers["Authorization"] = "Bearer " + os.environ[self.key_env]
        request = urllib.request.Request(
            self.endpoint + "/" + operation, data=json.dumps(body, allow_nan=False).encode(), headers=headers
        )
        from worldline.runtime.http import http_json

        return http_json(request, time.monotonic() + 10, 1024 * 1024)

    def claim(self):
        result = self.call("claim")
        if result["granted"]:
            self.thread = threading.Thread(target=self._heartbeat, daemon=True)
            self.thread.start()
        return result

    def _heartbeat(self):
        while not self.stop.wait(self.seconds / 3):
            try:
                self.call("heartbeat")
            except Exception as error:
                self.error = f"coordinator heartbeat failed: {type(error).__name__}"
                return

    def complete(self, result):
        return self.call("complete", result=result)

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=11)


def server_for(coordinator, host, port, token=None):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if token and self.headers.get("Authorization") != "Bearer " + token:
                self.send_error(401)
                return
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 1024 * 1024:
                    raise ValueError("invalid request size")
                result = coordinator.call(self.path.removeprefix("/"), json.loads(self.rfile.read(size)))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
            except (ValueError, KeyError, TypeError):
                self.send_error(409, "coordinator request rejected")

        def log_message(self, *_):
            pass

    return ThreadingHTTPServer((host, port), Handler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8769)
    parser.add_argument("--api-key-env", default="WORLDLINE_COORDINATOR_KEY")
    args = parser.parse_args()
    token = os.environ.get(args.api_key_env)
    if args.host not in {"127.0.0.1", "localhost", "::1"} and not token:
        parser.error("remote coordinator requires an API key; terminate TLS in your deployment")
    args.database.parent.mkdir(parents=True, exist_ok=True)
    coordinator = Coordinator(args.database)
    server = server_for(coordinator, args.host, args.port, token)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        coordinator.db.close()


if __name__ == "__main__":
    main()
