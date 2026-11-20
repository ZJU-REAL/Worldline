"""Evaluate any registered environment without knowing its action schema."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable

from worldline.contracts import Agent
from worldline import __version__
from worldline.errors import AgentError, BudgetExceeded, InvalidAction, ModelAPIError
from worldline.results import ArtifactRef, NormalizedResult
from worldline.run import RunRecord, write_json, write_record, make_run_id
from worldline.runtime.process import AgentView, ProcessEnvironment
from worldline.tasks import load_native_task
from worldline.trace import TraceEvent, TraceWriter
from worldline.media import MediaStore, MessageHistory
from worldline.runtime.verdict import checked_verdict
from worldline.agents.config import runtime_configuration, comparison_configuration, comparison_settings
from worldline.runtime.tools import public_spec


def evaluate_native(
    task_dir: Path,
    *,
    agent_factory: Callable[..., Agent],
    agent_name: str,
    model: str | None = None,
    runs_dir: Path = Path("runs"),
    dry_run: bool = False,
    harness_version: str = "native-tools/0.2",
    agent_config: dict | None = None,
    on_created: Callable | None = None,
    logical_trial_id: str | None = None,
    attempt: int = 1,
    worker_env: dict[str, str] | None = None,
    cancel_check: Callable | None = None,
) -> tuple[RunRecord, Path]:
    definition = load_native_task(task_dir)
    task = definition.task
    run_id = make_run_id(task.id)
    run_dir = runs_dir.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    record = RunRecord(
        run_id,
        task.id,
        task.version,
        task.environment,
        agent_name,
        model,
        (),
        "planned" if dry_run else "created",
        metadata={
            **task.metadata,
            "harness_version": harness_version,
            "worldline_version": __version__,
            "agent_config": agent_config or {},
            "environment_config": dict(definition.environment_config),
            "seed": task.seed,
            "split": task.split,
            "evaluation": {
                "timeout_sec": definition.timeout_sec,
                "max_actions": definition.max_actions,
                "max_turns": definition.max_turns,
                "reset_budget": definition.reset_budget,
                "restore_budget": definition.restore_budget,
                "fork_budget": definition.fork_budget,
            },
        },
    )
    write_record(run_dir, record)
    if on_created is not None:
        on_created(record, run_dir)
    write_json(run_dir / "task.json", asdict(task))
    media = MediaStore(run_dir)
    trace = TraceWriter(run_dir / "trace.jsonl", sync_every=32, media_store=media)

    def emit(kind, payload):
        trace.append(TraceEvent(run_id=run_id, kind=kind, payload=payload))

    emit("run.created", {"backend": task.environment})
    if dry_run:
        emit("run.planned", {})
        trace.close()
        return record, run_dir
    record = replace(record, status="running", started_at_ns=time.time_ns())
    write_record(run_dir, record)
    environment = None
    view = None
    messages = MessageHistory(media)
    messages.append({"role": "user", "content": task.instruction})
    usage = {}
    outcome, reward, error, metrics = "infra_error", None, None, {}
    error_code, retryable, interrupted = None, False, None
    environment_spec_hash = None
    runtime_config = {}
    timings = {}
    provider_failure = None
    try:
        stage = time.monotonic()
        environment = (
            ProcessEnvironment(definition, run_dir, worker_env=worker_env)
            if worker_env
            else ProcessEnvironment(definition, run_dir)
        )
        if cancel_check:
            environment.start_cancellation_monitor(cancel_check)
        initial = environment.start()
        record = replace(record, metadata={**record.metadata, "execution": environment.execution})
        write_record(run_dir, record)
        timings["environment_startup_s"] = time.monotonic() - stage
        from worldline.agents.tool_agent import tools_for

        environment.spec["tools"] = tools_for(environment.spec)
        environment_spec_hash = hashlib.sha256(json.dumps(environment.spec, sort_keys=True).encode()).hexdigest()
        emit("environment.started", {"spec": environment.spec, "observation": initial})
        view = AgentView(environment, emit)
        agent = agent_factory(
            spec=public_spec(environment.spec),
            initial_observation=initial,
            emit=emit,
            messages=messages,
            usage=usage,
            max_turns=definition.max_turns,
            deadline=environment.deadline,
        )
        runtime_config = runtime_configuration(agent)
        if cancel_check and hasattr(agent, "cancel_check"):
            agent.cancel_check = cancel_check
        json.dumps(runtime_config, allow_nan=False)
        record = replace(
            record,
            model=record.model or runtime_config["settings"].get("model"),
            metadata={**record.metadata, "agent_runtime_config": runtime_config},
        )
        write_record(run_dir, record)
        try:
            stage = time.monotonic()
            agent.run(task, view)
        except (BudgetExceeded, InvalidAction, AgentError) as agent_error:
            error = f"{type(agent_error).__name__}: {agent_error}"
            emit("agent.stopped", {"error": error})
        except ModelAPIError as api_error:
            provider_failure = api_error
            error, error_code, retryable = str(api_error), api_error.code, api_error.retryable
            emit("agent.provider_error", {"error_code": error_code, "retryable": retryable})
        finally:
            timings["agent_duration_s"] = time.monotonic() - stage
            view.close()
        stage = time.monotonic()
        if cancel_check and cancel_check():
            from worldline.errors import EnvironmentError

            raise EnvironmentError("run was cancelled by its lease owner", code="run.cancelled")
        verdict = environment.verify(task, run_dir)
        timings["verification_duration_s"] = time.monotonic() - stage
        outcome, reward, metrics = checked_verdict(verdict)
        success = outcome == "success"
        if provider_failure and not success and metrics.get("terminated") != 1:
            outcome, reward = "infra_error", None
        elif outcome != "infra_error":
            retryable = False
    except BaseException as infra:
        error = f"{type(infra).__name__}: {infra}"
        error_code = getattr(
            infra, "code", "run.interrupted" if isinstance(infra, KeyboardInterrupt) else "run.permanent_error"
        )
        retryable = bool(getattr(infra, "retryable", False))
        if not isinstance(infra, Exception):
            interrupted = infra
        emit("run.infra_error", {"error": error})
    finally:
        if view is not None:
            view.close()
            metrics.update(n_actions=float(view.actions), n_resets=float(view.resets))
        if environment is not None:
            metrics.update(getattr(environment, "timings", {}))
            environment.close()
        write_json(
            run_dir / "agent" / "trajectory.json",
            {
                "schema_version": "worldline.messages/0.2",
                "session_id": run_id,
                "messages": media.externalize(messages),
                "tools": environment.spec.get("tools", []) if environment is not None else [],
            },
        )
    media_files, media_bytes = media.inventory()
    metrics.update(
        run_duration_s=(time.time_ns() - record.started_at_ns) / 1e9,
        media_files=float(media_files),
        media_bytes=float(media_bytes),
    )
    metrics.update(timings)
    emit("run.finished", {"outcome": outcome, "reward": reward, "error": error})
    artifacts = []
    for kind, relative in (
        ("agent_trace", "agent/trajectory.json"),
        ("verdict_evidence", "verifier/final.json"),
        ("environment_journal", "trace.jsonl"),
    ):
        path = run_dir / relative
        if path.is_file():
            artifacts.append(ArtifactRef(kind, relative, path.stat().st_size))
    configuration = {
        "task_version": task.version,
        "seed": task.seed,
        "backend": task.environment,
        "budgets": record.metadata["evaluation"],
        "harness": harness_version,
        "worldline_version": __version__,
        "agent_config": comparison_settings(agent_config or {}),
        "agent_runtime_config": comparison_configuration(runtime_config) if runtime_config else {},
        "environment_spec_hash": environment_spec_hash,
    }
    key = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    result = NormalizedResult(
        "worldline.result/0.1",
        task.environment,
        run_id,
        task.id,
        outcome,
        reward,
        metrics=metrics,
        usage=usage,
        artifacts=tuple(artifacts),
        source=str(run_dir),
        error=error,
        task_version=task.version,
        comparison_key=f"{task.id}@{key}",
        agent=agent_name,
        model=record.model,
        error_code=error_code,
        retryable=retryable,
        logical_trial_id=logical_trial_id,
        attempt=attempt,
    )
    write_json(run_dir / "result.normalized.json", asdict(result))
    record = replace(
        record,
        status="infra_error" if outcome == "infra_error" else "completed",
        finished_at_ns=time.time_ns(),
        exit_code=2 if outcome == "infra_error" else 0,
    )
    write_record(run_dir, record)
    trace.close()
    if interrupted is not None:
        raise interrupted
    return record, run_dir
