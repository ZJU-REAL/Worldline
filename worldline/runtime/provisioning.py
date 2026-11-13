"""Owned command sessions and exclusive, identity-checked external sessions."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

from worldline.errors import EnvironmentError
from worldline.run import write_json
from worldline.storage import publish_bytes


def state_root():
    return Path(os.environ.get("WORLDLINE_STATE_DIR", Path.home() / ".cache/worldline")).resolve()


class ExternalSession:
    """A provider stays in the worker's process group and owns its descendants.

    It writes a ready descriptor only after provisioning, then handles a close
    message on stdin. No shell expansion and no task-specific simulator imports.
    Borrowed sessions are single-use unless their owner supplies a new identity.
    """

    def __init__(self, task, config):
        self.process = self.log = self.lock = None
        self.closed = False
        self.owned = bool(config.get("provider"))
        self.directory = Path(os.environ.get("WORLDLINE_RUN_DIR", state_root() / "sessions")) / uuid.uuid4().hex
        self.directory.mkdir(parents=True, exist_ok=False)
        expected = {
            "task_id": task.id,
            "task_version": task.version,
            "seed": task.seed,
            "source_revision": config["source_revision"],
        }
        for field in ("task_config_sha256", "image_digest", "controller_revision"):
            if field in config:
                expected[field] = config[field]
        try:
            legacy = False
            if self.owned:
                expected["session_id"] = uuid.uuid4().hex
                provider = config["provider"]
                command = provider.get("command")
                if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
                    raise ValueError("provider.command must be a nonempty argument array")
                ready = self.directory / "ready.json"
                request = self.directory / "request.json"
                write_json(request, {"schema_version": "worldline.session/0.1", **expected})
                replacements = {
                    "request_file": str(request),
                    "ready_file": str(ready),
                    "session_root": str(self.directory),
                }
                argv = [arg.format_map(replacements) for arg in command]
                self.log = (self.directory / "provider.log").open("wb")
                self.process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=self.log, stderr=self.log)
                timeout = provider.get("startup_timeout_sec", 60)
                if type(timeout) not in {int, float} or not 0 < timeout <= 3600:
                    raise ValueError("provider startup timeout must be in (0, 3600]")
                deadline = time.monotonic() + timeout
                while not ready.exists():
                    if self.process.poll() is not None:
                        raise EnvironmentError("session provider exited before readiness")
                    if time.monotonic() >= deadline:
                        raise EnvironmentError(
                            "session provider readiness timed out", code="session.start_timeout", retryable=True
                        )
                    time.sleep(0.05)
                self.descriptor = self._read(ready)
            elif config.get("session_manifest"):
                self.descriptor = self._read(Path(config["session_manifest"]))
            elif config.get("allow_unverified_session") is True:
                legacy = True
                self.descriptor = {
                    **config,
                    "session_id": config.get("session_id", "legacy"),
                    "identity_verified": False,
                }
            else:
                raise ValueError(
                    "external session requires provider or session_manifest; legacy use requires allow_unverified_session=true"
                )
            self.verified = not legacy
            if self.verified and self.descriptor.get("identity_verified") is False:
                raise EnvironmentError("session descriptor cannot disable identity verification")
            if self.verified:
                if self.descriptor.get("schema_version") != "worldline.session/0.1":
                    raise ValueError("unsupported session descriptor")
                for field, value in expected.items():
                    if type(self.descriptor.get(field)) is not type(value) or self.descriptor.get(field) != value:
                        raise EnvironmentError(f"session {field} mismatch")
                if not isinstance(self.descriptor.get("session_id"), str) or not self.descriptor["session_id"]:
                    raise EnvironmentError("session_id is required")
            endpoint = self.descriptor.get("socket_path") or self.descriptor.get("endpoint")
            if not isinstance(endpoint, str) or not endpoint:
                raise ValueError("session descriptor requires socket_path or endpoint")
            identity = str(Path(endpoint).resolve()) if "socket_path" in self.descriptor else endpoint.rstrip("/")
            locks = state_root() / "session-locks"
            locks.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256(identity.encode()).hexdigest()
            self.lock = (locks / f"{digest}.lock").open("a")
            try:
                fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise EnvironmentError("session endpoint is already owned by another trial") from None
            marker = locks / f"{digest}.used.json"
            session_hash = hashlib.sha256(self.descriptor["session_id"].encode()).hexdigest()
            consumed = locks / "consumed" / digest / f"{session_hash}.json"
            receipt = {"session_id": self.descriptor["session_id"], "task_id": task.id}
            if not publish_bytes(consumed, json.dumps(receipt).encode(), exclusive=True):
                raise EnvironmentError("external session has already been used; provision a fresh session")
            write_json(marker, receipt)
            write_json(
                self.directory / "session.json",
                {**self.descriptor, "owned": self.owned, "identity_verified": self.verified},
            )
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _read(path):
        with path.open("rb") as stream:
            raw = stream.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise EnvironmentError("session descriptor exceeds 1 MiB")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("session descriptor must be an object")
        return value

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.process is not None:
            if self.process.poll() is None:
                try:
                    self.process.stdin.write(b'{"method":"close"}\n')
                    self.process.stdin.flush()
                    self.process.wait(timeout=0.25)
                except (OSError, subprocess.TimeoutExpired):
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=0.25)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait(timeout=2)
            self.process.stdin.close()
        if self.log:
            self.log.close()
        if self.lock:
            self.lock.close()
