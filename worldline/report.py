"""Aggregate comparable evaluation summaries from normalized results."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

from worldline.datasets import discover_results
from worldline.run import write_json


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _sum_numeric(records: Iterable[Mapping[str, Any]], field: str) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for record in records:
        values = record.get(field)
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[key] += float(value)
    return dict(sorted(totals.items()))


def _summary(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    outcomes = {
        name: sum(record.get("outcome") == name for record in records) for name in ("success", "failure", "infra_error")
    }
    counted = outcomes["success"] + outcomes["failure"]
    interval = None
    if counted:
        z = 1.959963984540054
        proportion = outcomes["success"] / counted
        center = (proportion + z * z / (2 * counted)) / (1 + z * z / counted)
        margin = (
            z
            * math.sqrt(proportion * (1 - proportion) / counted + z * z / (4 * counted * counted))
            / (1 + z * z / counted)
        )
        interval = [max(0, center - margin), min(1, center + margin)]
    rewards = [
        float(record["reward"])
        for record in records
        if record.get("outcome") in {"success", "failure"} and isinstance(record.get("reward"), (int, float))
    ]
    return {
        "trials": len(records),
        "counted_trials": counted,
        "outcomes": outcomes,
        "success_rate": outcomes["success"] / counted if counted else None,
        "success_rate_wilson_95": interval,
        "infra_error_rate": outcomes["infra_error"] / len(records) if records else None,
        "mean_reward": fmean(rewards) if rewards else None,
        "usage_totals": _sum_numeric(records, "usage"),
        "metric_totals": _sum_numeric(records, "metrics"),
    }


def build_report(inputs: Sequence[Path]) -> dict[str, Any]:
    batch_summaries = {}
    for input_path in inputs:
        if input_path.is_dir():
            for path in input_path.rglob("batches/*/manifest.json"):
                manifest = _load(path)
                if str(manifest.get("schema_version", "")).startswith("worldline.batch/"):
                    statuses = manifest.get("statuses", {})
                    batch_summaries[manifest.get("batch_id", str(path))] = {
                        "tasks": manifest.get("tasks"),
                        "statuses": statuses,
                        "unscored_jobs": sum(value for key, value in statuses.items() if key != "completed"),
                    }
    records = [_load(path) for path in discover_results(inputs)]
    records = [record for record in records if record]
    superseded = sum(record.get("counted") is False for record in records)
    seen = set()
    canonical = []
    attempts = []
    for record in records:
        identity = (record.get("trial_id"), record.get("source"))
        if all(identity) and identity in seen:
            continue
        if all(identity):
            seen.add(identity)
        attempts.append(record)
        if record.get("counted") is False:
            continue
        canonical.append(record)
    records = canonical
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_configuration: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_agent: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    evaluations: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_task[str(record.get("task_id") or "unknown")].append(record)
        by_configuration[
            str(record.get("comparison_key") or f"{record.get('task_id') or 'unknown'}@unversioned")
        ].append(record)
        by_agent[f"{record.get('agent') or 'unknown'}@{record.get('model') or 'unspecified'}"].append(record)
        evaluations[
            f"{record.get('comparison_key') or record.get('task_id')}::{record.get('agent')}::{record.get('model')}"
        ].append(record)
    mixed_tasks = {
        task: sorted({str(record.get("comparison_key") or f"{task}@unversioned") for record in task_records})
        for task, task_records in by_task.items()
        if len({str(record.get("comparison_key") or f"{task}@unversioned") for record in task_records}) > 1
    }
    return {
        "schema_version": "worldline.report/0.1",
        "inputs": [str(path.resolve()) for path in inputs],
        "summary": _summary(records),
        "superseded_attempts": superseded,
        "all_attempt_usage_totals": _sum_numeric(attempts, "usage"),
        "batches": batch_summaries,
        "evaluations": {key: _summary(group) for key, group in sorted(evaluations.items())},
        "tasks": {task: _summary(task_records) for task, task_records in sorted(by_task.items())},
        "configurations": {key: _summary(config_records) for key, config_records in sorted(by_configuration.items())},
        "agents": {key: _summary(agent_records) for key, agent_records in sorted(by_agent.items())},
        "warnings": {"mixed_task_configurations": mixed_tasks},
    }


def write_report(inputs: Sequence[Path], output: Path) -> dict[str, Any]:
    report = build_report(inputs)
    write_json(output.resolve(), report)
    return report
