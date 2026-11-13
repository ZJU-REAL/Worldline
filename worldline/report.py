"""Aggregate comparable evaluation summaries from normalized results."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from worldline.datasets import iter_result_paths
from worldline.data_index import DiskIndex
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
    accumulator = Accumulator()
    for record in records:
        accumulator.add(record)
    return accumulator.result()


class Accumulator:
    def __init__(self):
        self.trials = self.rewards = 0
        self.reward_sum = 0.0
        self.outcomes = {key: 0 for key in ("success", "failure", "infra_error")}
        self.usage, self.metrics = defaultdict(float), defaultdict(float)

    def add(self, record):
        self.trials += 1
        outcome = record.get("outcome")
        if outcome in self.outcomes:
            self.outcomes[outcome] += 1
        reward = record.get("reward")
        if outcome in {"success", "failure"} and type(reward) in {int, float} and math.isfinite(reward):
            self.rewards += 1
            self.reward_sum += reward
        for field, totals in (("usage", self.usage), ("metrics", self.metrics)):
            values = record.get(field) or {}
            if not isinstance(values, dict):
                continue
            for key, value in values.items():
                if type(value) in {int, float} and math.isfinite(value):
                    totals[key] += value

    def result(self):
        counted = self.outcomes["success"] + self.outcomes["failure"]
        interval, proportion = None, None
        if counted:
            z = 1.959963984540054
            proportion = self.outcomes["success"] / counted
            center = (proportion + z * z / (2 * counted)) / (1 + z * z / counted)
            margin = (
                z
                * math.sqrt(proportion * (1 - proportion) / counted + z * z / (4 * counted * counted))
                / (1 + z * z / counted)
            )
            interval = [max(0, center - margin), min(1, center + margin)]
        return {
            "trials": self.trials,
            "counted_trials": counted,
            "outcomes": dict(self.outcomes),
            "success_rate": proportion,
            "success_rate_wilson_95": interval,
            "infra_error_rate": self.outcomes["infra_error"] / self.trials if self.trials else None,
            "mean_reward": self.reward_sum / self.rewards if self.rewards else None,
            "usage_totals": dict(sorted(self.usage.items())),
            "metric_totals": dict(sorted(self.metrics.items())),
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
    summary, attempts = Accumulator(), Accumulator()
    by_task, by_configuration, by_agent, evaluations = (defaultdict(Accumulator) for _ in range(4))
    versions = defaultdict(set)
    superseded = 0
    with DiskIndex() as seen:
        for path in iter_result_paths(inputs):
            record = _load(path)
            if not record:
                continue
            superseded += record.get("counted") is False
            identity = (record.get("trial_id"), record.get("source"))
            if all(identity) and not seen.add(identity):
                continue
            attempts.add(record)
            if record.get("counted") is False:
                continue
            summary.add(record)
            task = str(record.get("task_id") or "unknown")
            configuration = str(record.get("comparison_key") or f"{task}@unversioned")
            versions[task].add(configuration)
            by_task[task].add(record)
            by_configuration[configuration].add(record)
            by_agent[f"{record.get('agent') or 'unknown'}@{record.get('model') or 'unspecified'}"].add(record)
            evaluations[
                f"{record.get('comparison_key') or record.get('task_id')}::{record.get('agent')}::{record.get('model')}"
            ].add(record)
    mixed_tasks = {task: sorted(keys) for task, keys in versions.items() if len(keys) > 1}
    return {
        "schema_version": "worldline.report/0.1",
        "inputs": [str(path.resolve()) for path in inputs],
        "summary": summary.result(),
        "superseded_attempts": superseded,
        "all_attempt_usage_totals": dict(sorted(attempts.usage.items())),
        "batches": batch_summaries,
        "evaluations": {key: group.result() for key, group in sorted(evaluations.items())},
        "tasks": {key: group.result() for key, group in sorted(by_task.items())},
        "configurations": {key: group.result() for key, group in sorted(by_configuration.items())},
        "agents": {key: group.result() for key, group in sorted(by_agent.items())},
        "warnings": {"mixed_task_configurations": mixed_tasks},
    }


def write_report(inputs: Sequence[Path], output: Path) -> dict[str, Any]:
    report = build_report(inputs)
    write_json(output.resolve(), report)
    return report
