"""Aggregate comparable evaluation summaries from normalized results."""

from __future__ import annotations

import json
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
    outcomes = {name: sum(record.get("outcome") == name for record in records) for name in ("success", "failure", "infra_error")}
    counted = outcomes["success"] + outcomes["failure"]
    rewards = [float(record["reward"]) for record in records if isinstance(record.get("reward"), (int, float))]
    return {
        "trials": len(records),
        "counted_trials": counted,
        "outcomes": outcomes,
        "success_rate": outcomes["success"] / counted if counted else None,
        "infra_error_rate": outcomes["infra_error"] / len(records) if records else None,
        "mean_reward": fmean(rewards) if rewards else None,
        "usage_totals": _sum_numeric(records, "usage"),
        "metric_totals": _sum_numeric(records, "metrics"),
    }


def build_report(inputs: Sequence[Path]) -> dict[str, Any]:
    records = [_load(path) for path in discover_results(inputs)]
    records = [record for record in records if record]
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_task[str(record.get("task_id") or "unknown")].append(record)
    return {
        "schema_version": "worldline.report/0.1",
        "inputs": [str(path.resolve()) for path in inputs],
        "summary": _summary(records),
        "tasks": {task: _summary(task_records) for task, task_records in sorted(by_task.items())},
    }


def write_report(inputs: Sequence[Path], output: Path) -> dict[str, Any]:
    report = build_report(inputs)
    write_json(output.resolve(), report)
    return report
