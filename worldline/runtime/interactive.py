"""Persistent native environment sessions for externally driven agent loops."""

from __future__ import annotations

import hashlib
import json
import time
import threading
from functools import wraps
from dataclasses import asdict, replace
from pathlib import Path

from worldline import __version__
from worldline.agents.tool_agent import observation_content, prepare_messages, tools_for
from worldline.errors import AgentError, BudgetExceeded, InvalidAction
from worldline.media import MediaStore, model_context
from worldline.results import ArtifactRef, NormalizedResult
from worldline.run import RunRecord, make_run_id, write_json, write_record
from worldline.runtime.process import AgentView, ProcessEnvironment
from worldline.runtime.tools import execute_tool, public_spec
from worldline.runtime.verdict import checked_verdict
from worldline.tasks import load_native_task
from worldline.trace import TraceEvent, TraceWriter


def _serialized(operation):
    @wraps(operation)
    def wrapped(self, *args, **kwargs):
        with self.lock:
            return operation(self, *args, **kwargs)

    return wrapped


class InteractiveSession:
    """Trusted controller; external policies get public tools, never verifier access.

    Use a context manager or explicitly call finish/abort. Each session creates
    a fresh worker, records all turns, and grades new actions on their final state.
    """

    def __init__(
        self,
        task_dir: Path,
        *,
        runs_dir=Path("runs"),
        agent_name="external-loop",
        model=None,
        harness_version="interactive-tools/0.1",
        agent_config=None,
        worker_env=None,
    ):
        self.definition = load_native_task(task_dir)
        self.lock = threading.RLock()
        self.task = self.definition.task
        self.run_id = make_run_id(self.task.id)
        self.run_dir = runs_dir.resolve() / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.record = RunRecord(
            self.run_id,
            self.task.id,
            self.task.version,
            self.task.environment,
            agent_name,
            model,
            (),
            "created",
            metadata={
                **self.task.metadata,
                "seed": self.task.seed,
                "split": self.task.split,
                "worldline_version": __version__,
                "harness_version": harness_version,
                "agent_config": agent_config or {},
                "environment_config": dict(self.definition.environment_config),
                "evaluation": {
                    key: getattr(self.definition, key)
                    for key in ("timeout_sec", "max_actions", "max_turns", "reset_budget", "restore_budget")
                },
            },
        )
        self.media = MediaStore(self.run_dir)
        self.trace = TraceWriter(self.run_dir / "trace.jsonl", sync_every=32, media_store=self.media)
        self.worker_env = worker_env
        self.environment = self.view = self.result = None
        self.messages, self.usage, self.call_ids = [], {}, set()
        self.turns, self.done, self._closed = 0, False, False
        self.watchdog, self.watchdog_error, self.expired = None, None, False
        self.restores = 0
        self.checkpoint_ids = set()
        write_record(self.run_dir, self.record)
        write_json(self.run_dir / "task.json", asdict(self.task))
        self.emit("run.created", {"backend": self.task.environment})

    def emit(self, kind, payload):
        self.trace.append(TraceEvent(self.run_id, kind, payload))

    @_serialized
    def model_context(self, max_images=None):
        return model_context(self.messages, root=self.run_dir, max_images=max_images)

    @_serialized
    def start(self):
        if self.environment is not None or self.result is not None:
            raise RuntimeError("session was already started or closed")
        self.record = replace(self.record, status="running", started_at_ns=time.time_ns())
        write_record(self.run_dir, self.record)
        try:
            self.environment = ProcessEnvironment(self.definition, self.run_dir, worker_env=self.worker_env)
            initial = self.environment.start()
            self.record = replace(
                self.record, metadata={**self.record.metadata, "execution": self.environment.execution}
            )
            write_record(self.run_dir, self.record)
            self.environment.spec["tools"] = tools_for(self.environment.spec)
            self.view = AgentView(self.environment, self.emit)
            prepare_messages(self.messages, self.task, public_spec(self.environment.spec), initial)
            self.messages[:] = self.media.externalize(self.messages)
            self.emit("environment.started", {"spec": self.environment.spec, "observation": initial})
            self.watchdog = threading.Timer(max(0, self.environment.deadline - time.monotonic()), self._expire)
            self.watchdog.daemon = True
            self.watchdog.start()
            return {
                "spec": public_spec(self.environment.spec),
                "observation": initial,
                "deadline": self.environment.deadline,
                "run_id": self.run_id,
                "task": {
                    "id": self.task.id,
                    "instruction": self.task.instruction,
                    "version": self.task.version,
                    "seed": self.task.seed,
                },
            }
        except BaseException as error:
            self.abort(error)
            raise

    def _expire(self):
        self.expired = True
        try:
            with self.lock:
                if not self._closed:
                    self.finish()
        except Exception as error:
            self.watchdog_error = f"{type(error).__name__}: {error}"

    @_serialized
    def checkpoint(self):
        if self.view is None or self.done or self.result is not None:
            raise RuntimeError("session is not active")
        payload = {
            "schema_version": "worldline.checkpoint/0.1",
            "task_id": self.task.id,
            "task_version": self.task.version,
            "state": self.environment.checkpoint(),
        }
        identity = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
        write_json(self.run_dir / "checkpoints" / f"{identity}.json", payload)
        self.checkpoint_ids.add(identity)
        self.emit("environment.checkpoint", {"sha256": identity})
        return payload

    @_serialized
    def restore(self, checkpoint):
        if self.view is None or self.done or self.result is not None:
            raise RuntimeError("session is not active")
        if self.restores >= self.definition.restore_budget:
            raise BudgetExceeded("checkpoint restore budget exhausted")
        if (
            checkpoint.get("schema_version") != "worldline.checkpoint/0.1"
            or checkpoint.get("task_id") != self.task.id
            or checkpoint.get("task_version") != self.task.version
        ):
            raise ValueError("checkpoint task identity/version mismatch")
        identity = hashlib.sha256(json.dumps(checkpoint, sort_keys=True, allow_nan=False).encode()).hexdigest()
        if identity not in self.checkpoint_ids:
            raise ValueError("checkpoint was not issued by this session or was modified")
        self.restores += 1
        observation = self.environment.restore(checkpoint["state"])
        content = observation_content(observation)
        notice = "The trusted controller restored a checkpoint; runtime budgets remain cumulative."
        content = [{"type": "text", "text": notice}, *content] if isinstance(content, list) else notice + "\n" + content
        self.messages.append(self.media.externalize({"role": "user", "content": content}))
        self.emit("environment.restored", {"n_restores": self.restores, "observation": observation})
        return observation

    @_serialized
    def advance(self, message: dict) -> list[dict]:
        if self.expired or self.environment is not None and time.monotonic() >= self.environment.deadline:
            raise BudgetExceeded("session wall-clock budget exhausted")
        if self.view is None or self.done or self.result is not None:
            raise RuntimeError("session is not accepting policy turns")
        if self.turns >= self.definition.max_turns:
            self.done = True
            raise BudgetExceeded("model turn budget exhausted")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            raise AgentError("policy must supply an assistant message")
        message = {key: message[key] for key in ("role", "content", "tool_calls") if key in message}
        calls = message.get("tool_calls") or []
        if not isinstance(calls, list) or len(calls) > 32:
            raise AgentError("invalid or excessive tool calls")
        parsed, new_ids = [], set()
        for call in calls:
            try:
                ident, function = call["id"], call["function"]
                arguments = json.loads(function["arguments"])
                if (
                    not isinstance(ident, str)
                    or not ident
                    or ident in self.call_ids
                    or ident in new_ids
                    or not isinstance(arguments, dict)
                ):
                    raise ValueError("invalid or duplicate tool call id")
                new_ids.add(ident)
                parsed.append((ident, function["name"], arguments))
            except (KeyError, TypeError, ValueError) as error:
                raise AgentError(f"malformed tool call: {error}") from error
        self.call_ids.update(new_ids)
        self.turns += 1
        self.messages.append(self.media.externalize(message))
        self.emit("agent.message", {"message": message})
        responses, budget_error = [], None
        if not parsed:
            self.done = True
        for ident, name, arguments in parsed:
            try:
                if self.done:
                    raise InvalidAction("attempt already finished")
                observation, submitted = execute_tool(name, arguments, self.view)
                self.done = submitted
                content = observation_content(observation)
            except InvalidAction as error:
                content = json.dumps({"error": str(error)})
            except BudgetExceeded as error:
                budget_error, self.done = error, True
                content = json.dumps({"error": str(error)})
            except BaseException as error:
                self.abort(error)
                raise
            response = {"role": "tool", "tool_call_id": ident, "content": content}
            self.messages.append(self.media.externalize(response))
            self.emit("agent.tool_result", {"tool": name, "tool_call_id": ident, "content": content})
            responses.append(response)
        if budget_error is not None:
            raise budget_error
        return responses

    @_serialized
    def finish(self) -> NormalizedResult:
        if self.result is not None:
            return self.result
        if self.view is None:
            raise RuntimeError("session is not started")
        self.view.close()
        self.done = True
        try:
            verdict = self.environment.verify(self.task, self.run_dir)
            outcome, reward, metrics = checked_verdict(verdict)
        except BaseException as error:
            result = self.abort(error)
            if not isinstance(error, Exception):
                raise
            return result
        return self._publish(outcome, reward, metrics)

    @_serialized
    def abort(self, error=None) -> NormalizedResult:
        if self.result is not None:
            return self.result
        self.done = True
        if self.view is not None:
            self.view.close()
        return self._publish(
            "infra_error", None, {}, error=error or RuntimeError("external controller abandoned session")
        )

    def _publish(self, outcome, reward, metrics, *, error=None):
        try:
            if self.watchdog is not None:
                self.watchdog.cancel()
            if self.environment is not None:
                try:
                    self.environment.close()
                except Exception as cleanup_error:
                    outcome, reward, error = "infra_error", None, cleanup_error
            media_files, media_bytes = self.media.inventory()
            metrics.update(media_files=float(media_files), media_bytes=float(media_bytes))
            metrics.update(
                n_actions=float(self.view.actions if self.view else 0),
                n_turns=float(self.turns),
                n_resets=float(self.view.resets if self.view else 0),
                task_deadline_expired=float(self.expired),
                n_restores=float(self.restores),
            )
            configuration = {
                "task_version": self.task.version,
                "seed": self.task.seed,
                "backend": self.task.environment,
                "budgets": self.record.metadata["evaluation"],
                "harness": self.record.metadata["harness_version"],
                "worldline_version": __version__,
                "agent_config": self.record.metadata["agent_config"],
                "environment_spec_hash": hashlib.sha256(
                    json.dumps(self.environment.spec if self.environment else {}, sort_keys=True).encode()
                ).hexdigest(),
            }
            key = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
            write_json(
                self.run_dir / "agent/trajectory.json",
                {
                    "schema_version": "worldline.messages/0.2",
                    "session_id": self.run_id,
                    "messages": self.messages,
                    "tools": self.environment.spec["tools"] if self.view else [],
                },
            )
            artifacts = tuple(
                ArtifactRef(kind, name, (self.run_dir / name).stat().st_size)
                for kind, name in (
                    ("agent_trace", "agent/trajectory.json"),
                    ("verdict_evidence", "verifier/final.json"),
                )
                if (self.run_dir / name).is_file()
            )
            artifacts += tuple(
                ArtifactRef("environment_checkpoint", str(path.relative_to(self.run_dir)), path.stat().st_size)
                for path in sorted((self.run_dir / "checkpoints").glob("*.json"))
            )
            result = NormalizedResult(
                "worldline.result/0.1",
                self.task.environment,
                self.run_id,
                self.task.id,
                outcome,
                reward,
                metrics=metrics,
                usage=self.usage,
                artifacts=artifacts,
                source=str(self.run_dir),
                error=f"{type(error).__name__}: {error}" if error else None,
                task_version=self.task.version,
                comparison_key=f"{self.task.id}@{key}",
                agent=self.record.agent,
                model=self.record.model,
                error_code=getattr(error, "code", "controller.aborted") if error else None,
                retryable=bool(getattr(error, "retryable", False)),
                task_metadata={**self.task.metadata, "seed": self.task.seed, "split": self.task.split},
            )
            write_json(self.run_dir / "result.normalized.json", asdict(result))
            self.record = replace(
                self.record,
                status="infra_error" if outcome == "infra_error" else "completed",
                finished_at_ns=time.time_ns(),
                exit_code=2 if outcome == "infra_error" else 0,
            )
            write_record(self.run_dir, self.record)
            self.emit("run.finished", {"outcome": outcome, "reward": reward})
            self.result = result
            return result
        finally:
            self._closed = True
            self.trace.close()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, kind, error, traceback):
        if self.result is None and not self._closed:
            if error is None or isinstance(error, (BudgetExceeded, AgentError, InvalidAction)):
                self.finish()
            else:
                self.abort(error)
