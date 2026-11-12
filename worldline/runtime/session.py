"""Evaluate any registered environment without knowing its action schema."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable

from worldline.contracts import Agent
from worldline.errors import AgentError, BudgetExceeded, InvalidAction
from worldline.results import ArtifactRef, NormalizedResult
from worldline.run import RunRecord, write_json, write_record, make_run_id
from worldline.runtime.process import AgentView, ProcessEnvironment
from worldline.tasks import load_native_task
from worldline.trace import TraceEvent, TraceWriter


def evaluate_native(
    task_dir: Path,
    *,
    agent_factory: Callable[..., Agent],
    agent_name: str,
    model: str | None = None,
    runs_dir: Path = Path("runs"),
    dry_run: bool = False,
    harness_version: str = "native-tools/0.1",
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
            "environment_config": dict(definition.environment_config),
            "seed": task.seed,
            "split": task.split,
            "evaluation": {
                "timeout_sec": definition.timeout_sec,
                "max_actions": definition.max_actions,
                "max_turns": definition.max_turns,
                "reset_budget": definition.reset_budget,
            },
        },
    )
    write_record(run_dir, record)
    write_json(run_dir / "task.json", asdict(task))
    trace = TraceWriter(run_dir / "trace.jsonl")

    def emit(kind, payload):
        trace.append(TraceEvent(run_id=run_id, kind=kind, payload=payload))

    emit("run.created", {"backend": task.environment})
    if dry_run:
        emit("run.planned", {})
        return record, run_dir
    record = replace(record, status="running", started_at_ns=time.time_ns())
    write_record(run_dir, record)
    environment = None
    view = None
    messages = [{"role": "user", "content": task.instruction}]
    usage = {}
    outcome, reward, error, metrics = "infra_error", None, None, {}
    try:
        environment = ProcessEnvironment(definition, run_dir)
        initial = environment.start()
        from worldline.agents.tool_agent import tools_for

        environment.spec["tools"] = tools_for(environment.spec)
        emit("environment.started", {"spec": environment.spec, "observation": initial})
        view = AgentView(environment, emit)
        agent = agent_factory(
            spec=environment.spec,
            initial_observation=initial,
            emit=emit,
            messages=messages,
            usage=usage,
            max_turns=definition.max_turns,
            deadline=environment.deadline,
        )
        try:
            agent.run(task, view)
        except (BudgetExceeded, InvalidAction, AgentError) as agent_error:
            error = f"{type(agent_error).__name__}: {agent_error}"
            emit("agent.stopped", {"error": error})
        finally:
            view.close()
        verdict = environment.verify(task, run_dir)
        if verdict.infra_error:
            raise RuntimeError(verdict.infra_error)
        if not math.isfinite(verdict.reward):
            raise RuntimeError("verifier returned a non-finite reward")
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in verdict.metrics.values()):
            raise RuntimeError("verifier returned invalid metrics")
        reward, metrics = verdict.reward, dict(verdict.metrics)
        success = metrics.get("success", reward == 1) == 1
        outcome = "success" if success else "failure"
    except Exception as infra:
        error = f"{type(infra).__name__}: {infra}"
        emit("run.infra_error", {"error": error})
    finally:
        if view is not None:
            view.close()
            metrics.update(n_actions=float(view.actions), n_resets=float(view.resets))
        if environment is not None:
            environment.close()
        write_json(
            run_dir / "agent" / "trajectory.json",
            {
                "schema_version": "worldline.messages/0.1",
                "session_id": run_id,
                "messages": messages,
                "tools": environment.spec.get("tools", []) if environment is not None else [],
            },
        )
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
        model=model,
    )
    write_json(run_dir / "result.normalized.json", asdict(result))
    record = replace(
        record,
        status="infra_error" if outcome == "infra_error" else "completed",
        finished_at_ns=time.time_ns(),
        exit_code=2 if outcome == "infra_error" else 0,
    )
    write_record(run_dir, record)
    return record, run_dir
