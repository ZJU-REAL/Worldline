"""Training-data exports from verified Worldline results."""

from __future__ import annotations

import itertools
import json
import hashlib
import math
import sqlite3
import tempfile
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Mapping, Sequence

from worldline.run import write_json
from worldline.media import hydrate_media
from worldline.adapters.atif import atif_messages

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
    task_version: str = "unversioned"
    comparison_key: str = ""
    split: str = "unspecified"
    split_group: str = ""
    seed: int = 0
    tools: tuple[Mapping[str, Any], ...] = ()


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


def _trajectory_path(result: Mapping[str, Any]) -> Path | None:
    source = Path(str(result.get("source") or "")).resolve()
    for artifact in result.get("artifacts") or []:
        if isinstance(artifact, dict) and artifact.get("kind") == "agent_trace":
            try:
                candidate = (source / str(artifact.get("path"))).resolve(strict=True)
            except (OSError, RuntimeError):
                continue
            if candidate.is_relative_to(source) and candidate.is_file():
                return candidate
    return None


def iter_episodes(inputs: Sequence[Path], stats: dict[str, int]) -> Iterator[Episode]:
    stats.update(
        {
            "results": 0,
            "loaded": 0,
            "missing_trajectory": 0,
            "empty_trajectory": 0,
            "infra_error": 0,
            "held_out": 0,
            "invalid_verdict": 0,
            "superseded": 0,
            "duplicates": 0,
            "invalid_trajectory": 0,
        }
    )
    seen = set()
    for result_path in discover_results(inputs):
        stats["results"] += 1
        result = _load_object(result_path)
        if result.get("counted") is False:
            stats["superseded"] += 1
            continue
        identity = (result.get("trial_id"), result.get("source"))
        if identity in seen:
            stats["duplicates"] += 1
            continue
        seen.add(identity)
        if result.get("outcome") == "infra_error":
            stats["infra_error"] += 1
            continue
        if result.get("outcome") not in {"success", "failure"}:
            stats["invalid_verdict"] += 1
            continue
        task = _load_object(result_path.parent / "task.json")
        run = _load_object(result_path.parent / "run.json")
        metadata = {
            **(run.get("metadata") or {}),
            **(task.get("metadata") or {}),
            **(result.get("task_metadata") or {}),
        }
        declared_split = task.get("split", metadata.get("split", "unspecified"))
        if declared_split in {"test", "evaluation", "held_out"}:
            stats["held_out"] += 1
            continue
        trajectory_path = _trajectory_path(result)
        if trajectory_path is None:
            stats["missing_trajectory"] += 1
            continue
        trajectory = _load_object(trajectory_path)
        try:
            if trajectory.get("schema_version") in {"worldline.messages/0.1", "worldline.messages/0.2"}:
                messages = tuple(hydrate_media(trajectory.get("messages") or [], Path(result["source"])))
            else:
                messages = atif_messages(
                    trajectory, artifact_root=Path(result["source"]), trajectory_path=trajectory_path
                )
            if any(
                not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant", "tool"}
                for message in messages
            ):
                raise ValueError("invalid message")
        except (OSError, ValueError, KeyError, TypeError):
            stats["invalid_trajectory"] += 1
            continue
        if not messages:
            stats["empty_trajectory"] += 1
            continue
        reward = result.get("reward")
        if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
            stats["invalid_verdict"] += 1
            continue
        stats["loaded"] += 1
        yield Episode(
            id=str(result.get("trial_id") or trajectory.get("session_id") or result_path.parent.name),
            task_id=str(result.get("task_id") or "unknown"),
            outcome=str(result.get("outcome") or "failure"),
            reward=float(reward) if isinstance(reward, (int, float)) else 0.0,
            messages=messages,
            result_path=str(result_path),
            trajectory_path=str(trajectory_path),
            metrics=result.get("metrics") if isinstance(result.get("metrics"), dict) else {},
            usage=result.get("usage") if isinstance(result.get("usage"), dict) else {},
            task_version=str(result.get("task_version") or "unversioned"),
            comparison_key=str(result.get("comparison_key") or ""),
            split=declared_split,
            split_group=str(metadata.get("split_group") or result.get("task_id") or "unknown"),
            seed=metadata.get("seed", task.get("seed", 0)),
            tools=tuple(trajectory.get("tools") or []),
        )


def load_episodes(inputs: Sequence[Path]) -> tuple[list[Episode], dict[str, int]]:
    stats: dict[str, int] = {}
    return list(iter_episodes(inputs, stats)), stats


def _sft(episodes: Iterable[Episode], include_failures: bool) -> Iterator[dict[str, Any]]:
    for episode in episodes:
        if episode.outcome != "success" and not include_failures:
            continue
        yield {
            "id": episode.id,
            "messages": episode.messages,
            "tools": episode.tools,
            "metadata": {
                "task_id": episode.task_id,
                "outcome": episode.outcome,
                "reward": episode.reward,
                "result_path": episode.result_path,
                "trajectory_path": episode.trajectory_path,
                "task_version": episode.task_version,
                "seed": episode.seed,
                "split": episode.split,
                "split_group": episode.split_group,
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
            "task_version": episode.task_version,
            "seed": episode.seed,
            "split": episode.split,
            "split_group": episode.split_group,
            "tools": episode.tools,
            "metrics": episode.metrics,
            "usage": episode.usage,
            "provenance": {"result": episode.result_path, "trajectory": episode.trajectory_path},
        }


class PreferenceSpool:
    """Disk-backed pairing: RAM scales with one chosen/rejected pair."""

    def __init__(self, path: Path):
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE episodes (group_key TEXT, id TEXT, outcome TEXT, value BLOB)")
        self.db.execute("CREATE INDEX pairing ON episodes(group_key, outcome, id)")

    def add(self, episode: Episode):
        prompt = tuple(itertools.takewhile(lambda m: m.get("role") != "assistant", episode.messages))
        key = (episode.task_id, episode.task_version, episode.seed, episode.comparison_key, prompt)
        digest = hashlib.sha256(json.dumps(key, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        value = zlib.compress(json.dumps(asdict(episode), ensure_ascii=False).encode())
        self.db.execute("INSERT INTO episodes VALUES (?, ?, ?, ?)", (digest, episode.id, episode.outcome, value))

    def records(self, maximum: int):
        self.db.commit()
        for (group,) in self.db.execute("SELECT DISTINCT group_key FROM episodes ORDER BY group_key"):
            count = 0
            for (chosen,) in self.db.execute(
                "SELECT value FROM episodes WHERE group_key=? AND outcome='success' ORDER BY id", (group,)
            ):
                positive = json.loads(zlib.decompress(chosen))
                for (rejected,) in self.db.execute(
                    "SELECT value FROM episodes WHERE group_key=? AND outcome='failure' ORDER BY id", (group,)
                ):
                    if count >= maximum:
                        break
                    negative = json.loads(zlib.decompress(rejected))
                    count += 1
                    yield {
                        "id": f"{positive['task_id']}:{positive['id']}>{negative['id']}",
                        "task_id": positive["task_id"],
                        "chosen": positive["messages"],
                        "rejected": negative["messages"],
                        "chosen_reward": positive["reward"],
                        "rejected_reward": negative["reward"],
                        "task_version": positive["task_version"],
                        "seed": positive["seed"],
                        "split": positive["split"],
                        "split_group": positive["split_group"],
                        "tools": positive["tools"],
                        "provenance": {
                            "chosen_result": positive["result_path"],
                            "rejected_result": negative["result_path"],
                        },
                    }
                if count >= maximum:
                    break

    def close(self):
        self.db.close()


def export_datasets(
    inputs: Sequence[Path], outputs: Mapping[str, Path], *, include_failures=False, max_pairs_per_task=50
) -> dict[str, dict[str, Any]]:
    """Read each trajectory once and publish requested views after completion."""
    if not outputs or any(name not in {"sft", "rl", "preference"} for name in outputs):
        raise ValueError("unsupported export format")
    if max_pairs_per_task < 1:
        raise ValueError("max_pairs_per_task must be positive")
    outputs = {name: path.resolve() for name, path in outputs.items()}
    counts = {name: 0 for name in outputs}
    stats: dict[str, int] = {}
    streams, temporary = {}, {}
    with tempfile.TemporaryDirectory(prefix="worldline-pairs-") as directory:
        spool = PreferenceSpool(Path(directory) / "pairs.sqlite") if "preference" in outputs else None
        try:
            for name, output in outputs.items():
                output.parent.mkdir(parents=True, exist_ok=True)
                stream = tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=output.parent, prefix=f".{output.name}.", delete=False
                )
                streams[name], temporary[name] = stream, Path(stream.name)

            def append(name, record):
                streams[name].write(
                    json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
                )
                counts[name] += 1

            for episode in iter_episodes(inputs, stats):
                if "sft" in outputs:
                    for record in _sft([episode], include_failures):
                        append("sft", record)
                if "rl" in outputs:
                    append("rl", next(_rl([episode])))
                if spool is not None:
                    spool.add(episode)
            if spool is not None:
                for record in spool.records(max_pairs_per_task):
                    append("preference", record)
            for name, stream in streams.items():
                stream.close()
                temporary[name].replace(outputs[name])
        finally:
            if spool is not None:
                spool.close()
            for stream in streams.values():
                stream.close()
            for path in temporary.values():
                path.unlink(missing_ok=True)
    manifests = {}
    for name, output in outputs.items():
        manifest = {
            "schema_version": "worldline.dataset/0.1",
            "format": name,
            "records": counts[name],
            "inputs": [str(path.resolve()) for path in inputs],
            "load_stats": stats,
            "options": {"include_failures": include_failures, "max_pairs_per_task": max_pairs_per_task},
            "pipeline": "single-pass-streaming-with-disk-pairing",
        }
        write_json(output.with_suffix(output.suffix + ".manifest.json"), manifest)
        manifests[name] = manifest
    return manifests


def export_dataset(
    inputs: Sequence[Path], output: Path, *, format: ExportFormat, include_failures=False, max_pairs_per_task=50
) -> dict[str, Any]:
    return export_datasets(
        inputs, {format: output}, include_failures=include_failures, max_pairs_per_task=max_pairs_per_task
    )[format]


def _record_task_id(record: Mapping[str, Any]) -> str | None:
    task_id = record.get("task_id")
    if isinstance(task_id, str) and task_id:
        return task_id
    metadata = record.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("task_id"), str):
        return metadata["task_id"]
    return None


def _record_group(record: Mapping[str, Any]) -> str | None:
    return record.get("split_group") or (record.get("metadata") or {}).get("split_group") or _record_task_id(record)


def _record_split(record: Mapping[str, Any]) -> str:
    return record.get("split") or (record.get("metadata") or {}).get("split") or "unspecified"


def _jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(
                line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"non-finite JSON: {value}"))
            )
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: record must be a JSON object")
            yield line_number, value


def split_dataset(
    input_path: Path,
    output_dir: Path,
    *,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: str = "worldline-v1",
) -> dict[str, Any]:
    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(not math.isfinite(ratio) or ratio < 0 for ratio in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("train, validation, and test ratios must be non-negative and sum to 1")
    input_path = input_path.resolve()
    output_dir = output_dir.resolve()
    names = ("train", "validation", "test")
    counts = {name: 0 for name in names}
    tasks: dict[str, str] = {}
    declarations: dict[str, str] = {}
    for line_number, record in _jsonl(input_path):
        if not _record_task_id(record):
            raise ValueError(f"{input_path}:{line_number}: record has no task_id")
        group = _record_group(record)
        declared = _record_split(record)
        if declared in names:
            if group in declarations and declarations[group] != declared:
                raise ValueError(f"conflicting declared splits for group {group!r}")
            declarations[group] = declared
        elif declared in {"evaluation", "held_out"}:
            if group in declarations and declarations[group] != "test":
                raise ValueError(f"conflicting declared splits for group {group!r}")
            declarations[group] = "test"
    output_dir.mkdir(parents=True, exist_ok=True)
    streams = {name: (output_dir / f"{name}.jsonl").open("w", encoding="utf-8") for name in names}
    try:
        for line_number, record in _jsonl(input_path):
            task_id = _record_task_id(record)
            if not task_id:
                raise ValueError(f"{input_path}:{line_number}: record has no task_id")
            group = _record_group(record)
            digest = hashlib.sha256(f"{seed}\0{group}".encode()).digest()
            bucket = int.from_bytes(digest[:8], "big") / 2**64
            split = (
                "train" if bucket < train_ratio else "validation" if bucket < train_ratio + validation_ratio else "test"
            )
            split = declarations.get(group, split)
            tasks[task_id] = split
            original = _record_split(record)
            record["split"] = split
            if isinstance(record.get("metadata"), dict):
                record["metadata"] = {**record["metadata"], "split": split, "source_split": original}
            streams[split].write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            counts[split] += 1
    finally:
        for stream in streams.values():
            stream.close()
    manifest = {
        "schema_version": "worldline.split/0.1",
        "input": str(input_path),
        "seed": seed,
        "ratios": dict(zip(names, ratios, strict=True)),
        "records": counts,
        "tasks": {name: sum(split == name for split in tasks.values()) for name in names},
    }
    write_json(output_dir / "split.manifest.json", manifest)
    return manifest


def validate_datasets(paths: Sequence[Path]) -> dict[str, Any]:
    errors: list[str] = []
    seen_ids: dict[str, str] = {}
    tasks_by_file: dict[str, set[str]] = {}
    groups_by_file: dict[str, set[str]] = {}
    record_counts: dict[str, int] = {}
    for raw_path in paths:
        path = raw_path.resolve()
        tasks: set[str] = set()
        groups: set[str] = set()
        count = 0
        try:
            for line_number, record in _jsonl(path):
                count += 1
                record_id = record.get("id")
                if isinstance(record_id, str):
                    previous = seen_ids.get(record_id)
                    if previous:
                        errors.append(f"duplicate id {record_id!r}: {previous} and {path}:{line_number}")
                    else:
                        seen_ids[record_id] = f"{path}:{line_number}"
                task_id = _record_task_id(record)
                if not task_id:
                    errors.append(f"{path}:{line_number}: missing task_id")
                else:
                    tasks.add(task_id)
                group = _record_group(record)
                if group:
                    groups.add(group)
                for field in ("messages", "chosen", "rejected"):
                    if field in record and not isinstance(record[field], (list, tuple)):
                        errors.append(f"{path}:{line_number}: {field} must be an array")
        except (OSError, ValueError, json.JSONDecodeError) as error:
            errors.append(str(error))
        key = str(path)
        tasks_by_file[key] = tasks
        groups_by_file[key] = groups
        record_counts[key] = count

    overlaps: list[dict[str, Any]] = []
    for (left, left_tasks), (right, right_tasks) in itertools.combinations(tasks_by_file.items(), 2):
        shared = sorted(left_tasks & right_tasks)
        if shared:
            overlaps.append({"left": left, "right": right, "tasks": shared})
    if overlaps:
        errors.append(f"task overlap exists between {len(overlaps)} file pairs")
    group_overlaps = []
    for (left, left_groups), (right, right_groups) in itertools.combinations(groups_by_file.items(), 2):
        shared = sorted(left_groups & right_groups)
        if shared:
            group_overlaps.append({"left": left, "right": right, "groups": shared})
    if group_overlaps:
        errors.append(f"split group overlap exists between {len(group_overlaps)} file pairs")
    return {
        "schema_version": "worldline.validation/0.1",
        "valid": not errors,
        "records": record_counts,
        "unique_ids": len(seen_ids),
        "errors": errors,
        "overlaps": overlaps,
        "group_overlaps": group_overlaps,
    }
