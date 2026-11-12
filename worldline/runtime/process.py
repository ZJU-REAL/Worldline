"""Own a simulator process, enforce request deadlines, and reap it on every exit."""

from __future__ import annotations

import json
import os
import select
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from worldline.contracts import RunResult, SandboxCapabilities, Task
from worldline.errors import BudgetExceeded, EnvironmentError, InvalidAction
from worldline.tasks import NativeTask


class ProcessEnvironment:
    """Runner-owned channel. AgentView exposes only public interactions."""

    def __init__(self, definition: NativeTask, run_dir: Path, *, worker_env: dict[str, str] | None = None) -> None:
        self.definition = definition
        self.deadline = time.monotonic() + definition.timeout_sec
        self.log = (run_dir / "worker.log").open("wb")
        try:
            self.process = subprocess.Popen(
                [sys.executable, "-m", "worldline.runtime.worker"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.log,
                start_new_session=True,
                bufsize=0,
                env={**os.environ, **(worker_env or {})},
            )
        except BaseException:
            self.log.close()
            raise
        self.pending = b""
        self.spec: dict = {}
        self.deadline_abort = False

    def request(self, method: str, *, verification: bool = False, **payload: Any) -> Any:
        if self.process.poll() is not None:
            raise EnvironmentError(
                f"environment process exited with {self.process.returncode}; see worker.log",
                code="worker.deadline_abort" if self.deadline_abort else "worker.crashed",
                retryable=not self.deadline_abort,
            )
        deadline = time.monotonic() + 15 if verification else self.deadline
        if time.monotonic() >= deadline:
            raise BudgetExceeded("task wall-clock budget exhausted")
        encoded = json.dumps({"method": method, **payload}, allow_nan=False).encode() + b"\n"
        if len(encoded) > 4 * 1024 * 1024:
            raise InvalidAction("request exceeds 4 MiB")
        try:
            # A worker can hang before it reads a large request. Nonblocking
            # writes keep the same deadline effective during pipe backpressure.
            descriptor = self.process.stdin.fileno()
            os.set_blocking(descriptor, False)
            sent = 0
            while sent < len(encoded):
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([], [descriptor], [], max(0, remaining))[1]:
                    self.deadline_abort = True
                    self.close(graceful=False)
                    raise BudgetExceeded("worker request deadline exhausted")
                try:
                    sent += os.write(descriptor, encoded[sent:])
                except BlockingIOError:
                    continue
            while b"\n" not in self.pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([self.process.stdout], [], [], max(0, remaining))[0]:
                    self.deadline_abort = True
                    # A timed-out command may still mutate state. Kill it before grading.
                    self.close(graceful=False)
                    if verification:
                        raise EnvironmentError("verifier deadline exhausted")
                    raise BudgetExceeded("environment request exceeded the wall-clock budget")
                chunk = os.read(self.process.stdout.fileno(), 65536)
                if not chunk:
                    raise EnvironmentError(
                        "environment worker closed its response stream; see worker.log",
                        code="worker.crashed",
                        retryable=True,
                    )
                self.pending += chunk
                if len(self.pending) > 16 * 1024 * 1024:
                    raise EnvironmentError("environment response exceeds 16 MiB")
            line, self.pending = self.pending.split(b"\n", 1)
            response = json.loads(line)
        except (OSError, ValueError) as error:
            raise EnvironmentError(f"invalid worker response: {error}") from error
        if "error" in response:
            error = response["error"]
            cls = InvalidAction if error["type"] == "InvalidAction" else EnvironmentError
            raise cls(f"{error['type']}: {error['message']}")
        return response["result"]

    def start(self) -> Any:
        result = self.request(
            "start", task=asdict(self.definition.task), config=dict(self.definition.environment_config)
        )
        self.spec = result["spec"]
        return result["observation"]

    def verify(self, task: Task, run_dir: Path) -> RunResult:
        result = self.request("verify", verification=True)
        evidence = run_dir / "verifier" / "final.json"
        from worldline.run import write_json

        write_json(evidence, result)
        return RunResult(result["reward"], result.get("metrics", {}), (evidence,), result.get("infra_error"))

    def close(self, *, graceful: bool = True) -> None:
        if graceful and self.process.poll() is None:
            try:
                descriptor = self.process.stdin.fileno()
                os.set_blocking(descriptor, False)
                os.write(descriptor, b'{"method":"close"}\n')
                self.process.wait(timeout=0.5)
            except (OSError, subprocess.TimeoutExpired):
                pass
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=2)
        for stream in (self.process.stdin, self.process.stdout, self.log):
            if stream is not None:
                stream.close()


class AgentView:
    """Public sandbox operations, with task-declared interaction/reset limits.

    This is a tool boundary for hosted agents, not an OS sandbox for arbitrary
    Python plugins. Untrusted coding agents continue to use Harbor containers.
    """

    def __init__(self, environment: ProcessEnvironment, emit) -> None:
        self._environment = environment
        self._emit = emit
        self.actions = 0
        self.resets = 0
        self.active = True

    @property
    def capabilities(self) -> SandboxCapabilities:
        values = self._environment.spec.get("capabilities", {})
        return SandboxCapabilities(
            **{key: bool(values.get(key, False)) for key in SandboxCapabilities.__dataclass_fields__}
        )

    def _check(self) -> None:
        if not self.active:
            raise BudgetExceeded("agent session has ended")
        if time.monotonic() >= self._environment.deadline:
            raise BudgetExceeded("task wall-clock budget exhausted")

    def start(self, task: Task) -> Any:
        raise InvalidAction("the runner already started the environment")

    def observe(self) -> Any:
        self._check()
        result = self._environment.request("observe")
        self._emit("environment.observation", {"observation": result})
        return result

    def act(self, action: Any) -> Any:
        self._check()
        try:
            json.dumps(action, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise InvalidAction("action must be finite JSON data") from error
        if self.actions >= self._environment.definition.max_actions:
            raise BudgetExceeded("action budget exhausted")
        self.actions += 1
        self._emit("environment.action", {"action": action, "index": self.actions})
        result = self._environment.request("act", action=action)
        self._emit("environment.observation", {"observation": result})
        return result

    def reset(self) -> Any:
        self._check()
        if self.resets >= self._environment.definition.reset_budget:
            raise InvalidAction("task reset budget exhausted")
        self.resets += 1
        result = self._environment.request("reset", seed=self._environment.definition.task.seed)
        self._emit("environment.reset", {"seed": self._environment.definition.task.seed, "observation": result})
        return result

    def close(self) -> None:
        self.active = False
