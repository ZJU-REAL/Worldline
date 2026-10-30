"""Training-data exports from verified Worldline results."""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Mapping, Sequence

from worldline.run import write_json

ExportFormat = Literal["sft", "preference", "rl"]


@dataclass(frozen=True, slots=True)
class Episode:
    id: str
    task_id: str
    outcome: str
    reward: float
    messages: tuple[Mapping[str, Any], ...]
    result_path: str
    trajectory_path: str
    metrics: Mapping[str, float]
    usage: Mapping[str, float]


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def discover_results(inputs: Sequence[Path]) -> list[Path]:
    found: set[Path] = set()
    for input_path in inputs:
        path = input_path.resolve()
        if path.is_file() and path.name == "result.normalized.json":
            found.add(path)
        elif path.is_dir():
            found.update(path.rglob("result.normalized.json"))
    return sorted(found)


def _content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return json.dumps(value, ensure_ascii=False) if value is not None else ""
    parts: list[str] = []
    for part in value:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict):
            if isinstance(part.get("text"), str):
                parts.append(part["text"])
            elif part.get("type") in {"image", "audio"}:
                parts.append(f"[{part['type']}: {part.get('path') or part.get('url') or 'embedded'}]")
            else:
                parts.append(json.dumps(part, ensure_ascii=False))
    return "\n".join(part for part in parts if part)


def atif_messages(trajectory: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Convert ATIF steps to a chat/tool sequence without dropping tool observations."""

    messages: list[Mapping[str, Any]] = []
    role_map = {"agent": "assistant", "user": "user", "system": "system", "environment": "tool"}
    for step in trajectory.get("steps") or []:
        if not isinstance(step, dict):
            continue
        role = role_map.get(str(step.get("source")), str(step.get("source") or "user"))
        message: dict[str, Any] = {"role": role, "content": _content(step.get("message"))}
        tool_calls = step.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            message["tool_calls"] = tool_calls
        if message["content"] or message.get("tool_calls"):
            messages.append(message)

        observation = step.get("observation")
        if not isinstance(observation, dict):
            continue
        for result in observation.get("results") or []:
            if not isinstance(result, dict):
                continue
            tool_message: dict[str, Any] = {"role": "tool", "content": _content(result.get("content"))}
            source_call_id = result.get("source_call_id")
            if source_call_id:
                tool_message["tool_call_id"] = source_call_id
            if tool_message["content"] or source_call_id:
                messages.append(tool_message)
    return tuple(messages)


def _trajectory_path(result: Mapping[str, Any]) -> Path | None:
    source = Path(str(result.get("source") or ""))
    for artifact in result.get("artifacts") or []:
        if isinstance(artifact, dict) and artifact.get("kind") == "agent_trace":
            candidate = source / str(artifact.get("path"))
            if candidate.is_file():
                return candidate
    return None


def load_episodes(inputs: Sequence[Path]) -> tuple[list[Episode], dict[str, int]]:
    episodes: list[Episode] = []
    stats = {"results": 0, "loaded": 0, "missing_trajectory": 0, "empty_trajectory": 0, "infra_error": 0}
    for result_path in discover_results(inputs):
        stats["results"] += 1
        result = _load_object(result_path)
        if result.get("outcome") == "infra_error":
            stats["infra_error"] += 1
            continue
        trajectory_path = _trajectory_path(result)
        if trajectory_path is None:
            stats["missing_trajectory"] += 1
            continue
        trajectory = _load_object(trajectory_path)
        messages = atif_messages(trajectory)
        if not messages:
            stats["empty_trajectory"] += 1
            continue
        reward = result.get("reward")
        episodes.append(
            Episode(
                id=str(result.get("trial_id") or trajectory.get("session_id") or result_path.parent.name),
                task_id=str(result.get("task_id") or "unknown"),
                outcome=str(result.get("outcome") or "failure"),
                reward=float(reward) if isinstance(reward, (int, float)) else 0.0,
                messages=messages,
                result_path=str(result_path),
                trajectory_path=str(trajectory_path),
                metrics=result.get("metrics") if isinstance(result.get("metrics"), dict) else {},
                usage=result.get("usage") if isinstance(result.get("usage"), dict) else {},
            )
        )
        stats["loaded"] += 1
    return episodes, stats


def _sft(episodes: Iterable[Episode], include_failures: bool) -> Iterator[dict[str, Any]]:
    for episode in episodes:
        if episode.outcome != "success" and not include_failures:
            continue
        yield {
            "id": episode.id,
            "messages": episode.messages,
            "metadata": {
                "task_id": episode.task_id,
                "outcome": episode.outcome,
                "reward": episode.reward,
                "result_path": episode.result_path,
                "trajectory_path": episode.trajectory_path,
            },
        }


def _rl(episodes: Iterable[Episode]) -> Iterator[dict[str, Any]]:
    for episode in episodes:
        yield {
            "id": episode.id,
            "task_id": episode.task_id,
            "messages": episode.messages,
            "reward": episode.reward,
            "outcome": episode.outcome,
            "metrics": episode.metrics,
            "usage": episode.usage,
            "provenance": {"result": episode.result_path, "trajectory": episode.trajectory_path},
        }


def _preference(episodes: Iterable[Episode], max_pairs_per_task: int) -> Iterator[dict[str, Any]]:
    by_task: dict[str, list[Episode]] = {}
    for episode in episodes:
        by_task.setdefault(episode.task_id, []).append(episode)
    for task_id, task_episodes in sorted(by_task.items()):
        chosen = sorted((e for e in task_episodes if e.outcome == "success"), key=lambda e: e.id)
        rejected = sorted((e for e in task_episodes if e.outcome == "failure"), key=lambda e: e.id)
        for index, (positive, negative) in enumerate(itertools.product(chosen, rejected)):
            if index >= max_pairs_per_task:
                break
            yield {
                "id": f"{task_id}:{positive.id}>{negative.id}",
                "task_id": task_id,
                "chosen": positive.messages,
                "rejected": negative.messages,
                "chosen_reward": positive.reward,
                "rejected_reward": negative.reward,
                "provenance": {
                    "chosen_result": positive.result_path,
                    "rejected_result": negative.result_path,
                },
            }


def export_dataset(
    inputs: Sequence[Path],
    output: Path,
    *,
    format: ExportFormat,
    include_failures: bool = False,
    max_pairs_per_task: int = 50,
) -> dict[str, Any]:
    episodes, load_stats = load_episodes(inputs)
    if format == "sft":
        records = _sft(episodes, include_failures)
    elif format == "preference":
        records = _preference(episodes, max_pairs_per_task)
    elif format == "rl":
        records = _rl(episodes)
    else:
        raise ValueError(f"unsupported export format: {format}")

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    manifest = {
        "schema_version": "worldline.dataset/0.1",
        "format": format,
        "records": count,
        "inputs": [str(path.resolve()) for path in inputs],
        "load_stats": load_stats,
        "options": {"include_failures": include_failures, "max_pairs_per_task": max_pairs_per_task},
    }
    write_json(output.with_suffix(output.suffix + ".manifest.json"), manifest)
    return manifest
