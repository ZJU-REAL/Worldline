"""Normalize backend-specific trial outputs into stable Worldline results."""

from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from worldline.run import write_json
from worldline.trace import TraceEvent, TraceWriter


ARTIFACT_KINDS = {
    "trajectory.json": "agent_trace",
    "trajectory.npz": "environment_trajectory",
    "episode.jsonl": "environment_journal",
    "replay.mp4": "video",
    "live.mp4": "video",
    "replay.last.png": "image",
    "final.json": "verdict_evidence",
}


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    kind: str
    path: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class NormalizedResult:
    schema_version: str
    backend: str
    trial_id: str
    task_id: str | None
    outcome: str
    reward: float | None
    metrics: Mapping[str, float] = field(default_factory=dict)
    usage: Mapping[str, float] = field(default_factory=dict)
    artifacts: tuple[ArtifactRef, ...] = ()
    source: str = ""
    error: str | None = None
    task_version: str | None = None
    comparison_key: str | None = None
    agent: str | None = None
    model: str | None = None
    error_code: str | None = None
    retryable: bool = False
    counted: bool = True
    logical_trial_id: str | None = None
    attempt: int = 1
    superseded_by: str | None = None
    task_metadata: Mapping[str, Any] = field(default_factory=dict)


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _reward_txt(path: Path) -> float | None:
    try:
        return float(path.read_text(encoding="utf-8").split()[0])
    except (OSError, ValueError, IndexError):
        return None


def _numbers(values: Mapping[str, Any], *, exclude: set[str] | None = None) -> dict[str, float]:
    excluded = exclude or set()
    return {
        key: float(value)
        for key, value in values.items()
        if key not in excluded
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    }


def _artifacts(trial_dir: Path) -> tuple[ArtifactRef, ...]:
    found: list[ArtifactRef] = []
    trusted_root = trial_dir.resolve()
    for area in ("agent", "verifier", "artifacts"):
        root = trial_dir / area
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            try:
                resolved = path.resolve(strict=True)
                safe = resolved.is_relative_to(trusted_root)
            except (OSError, RuntimeError):
                safe = False
            if safe and path.is_file() and path.name in ARTIFACT_KINDS:
                found.append(
                    ArtifactRef(
                        kind=ARTIFACT_KINDS[path.name],
                        path=str(path.relative_to(trial_dir)),
                        size_bytes=path.stat().st_size,
                    )
                )
    return tuple(sorted(found, key=lambda artifact: (artifact.kind, artifact.path)))


def normalize_harbor_trial(trial_dir: Path, *, task_snapshot: Mapping[str, Any] | None = None) -> NormalizedResult:
    """Read a Harbor trial without mutating or copying its potentially large artifacts."""

    trial_dir = trial_dir.resolve()
    result = _load_object(trial_dir / "result.json")
    lock = _load_object(trial_dir / "lock.json")
    reward_path = trial_dir / "verifier" / "reward.json"
    verifier = _load_object(reward_path)
    reward = _reward_txt(trial_dir / "verifier" / "reward.txt")
    if reward is None and isinstance(verifier.get("success"), (int, float)):
        reward = float(verifier["success"])

    exception_info = result.get("exception_info")
    exception = exception_info.get("exception_type") if isinstance(exception_info, dict) else None
    infra = bool(verifier.get("infra_error") or verifier.get("grader_error"))
    invalid_reward = reward is not None and not math.isfinite(reward)
    success_value = verifier.get("success")
    invalid_success = success_value is not None and (
        not isinstance(success_value, (int, float)) or not math.isfinite(success_value)
    )
    contradictory = (
        isinstance(success_value, (int, float))
        and reward in {0, 1}
        and success_value in {0, 1}
        and reward != success_value
    )
    if not reward_path.is_file() or not verifier:
        infra = True
        error = "verifier wrote no readable reward.json"
    elif invalid_reward or invalid_success or contradictory:
        infra, reward = True, None
        error = "verifier wrote a non-finite or contradictory verdict"
    elif exception and exception != "AgentTimeoutError" and verifier.get("success") != 1:
        infra = True
        error = f"Harbor ended the trial with {exception}"
    elif infra:
        error = "verifier reported an infrastructure or grader error"
    else:
        error = None

    if infra:
        outcome = "infra_error"
    elif verifier.get("success") == 1 or reward == 1:
        outcome = "success"
    else:
        outcome = "failure"

    agent_result = result.get("agent_result") if isinstance(result.get("agent_result"), dict) else {}
    usage = _numbers(agent_result)
    gateway = _load_object(trial_dir / "gateway" / "totals.json")
    for key, value in _numbers(gateway).items():
        usage.setdefault(f"gateway_{key}", value)

    task_id = result.get("task_name")
    locked_task = lock.get("task") if isinstance(lock.get("task"), dict) else {}
    task_version = locked_task.get("digest") or result.get("task_checksum")
    agent_info = result.get("agent_info") if isinstance(result.get("agent_info"), dict) else {}
    model_info = agent_info.get("model_info") if isinstance(agent_info.get("model_info"), dict) else {}
    task_metadata = {}
    if task_snapshot:
        aliases = {task_snapshot["id"], task_snapshot["id"].rsplit("/", 1)[-1]}
        if task_snapshot.get("environment"):
            aliases.add(Path(task_snapshot["environment"]).name)
        if task_id and task_id not in aliases:
            raise ValueError("Harbor result task identity differs from the evaluation task snapshot")
        snapshot_version = task_snapshot["version"]
        if (
            task_version
            and snapshot_version != "unversioned"
            and str(task_version).removeprefix("sha256:") != str(snapshot_version).removeprefix("sha256:")
        ):
            raise ValueError("Harbor result version differs from the evaluation task snapshot")
        task_id = task_snapshot["id"]
        task_version = task_snapshot["version"]
        task_metadata = {
            **task_snapshot.get("metadata", {}),
            "seed": task_snapshot["seed"],
            "split": task_snapshot["split"],
        }
    return NormalizedResult(
        schema_version="worldline.result/0.1",
        backend="harbor",
        trial_id=str(result.get("id") or trial_dir.name),
        task_id=task_id,
        outcome=outcome,
        reward=reward,
        metrics=_numbers(verifier, exclude={"success", "infra_error", "grader_error"}),
        usage=usage,
        artifacts=_artifacts(trial_dir),
        source=str(trial_dir),
        error=error,
        task_version=str(task_version) if task_version else None,
        comparison_key=f"{task_id or 'unknown'}@{task_version or 'unversioned'}",
        agent=str(agent_info["name"]) if agent_info.get("name") else None,
        model=str(model_info["name"]) if model_info.get("name") else None,
        task_metadata=task_metadata,
    )


def ingest_harbor_trial(
    trial_dir: Path, runs_dir: Path = Path("runs"), *, task_dir: Path | None = None
) -> tuple[NormalizedResult, Path]:
    snapshot = None
    if task_dir is not None:
        from worldline.adapters.harbor import load_task

        snapshot = asdict(load_task(task_dir))
    result = normalize_harbor_trial(trial_dir, task_snapshot=snapshot)
    run_id = f"import-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:8]}"
    run_dir = runs_dir.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "result.normalized.json", asdict(result))
    if snapshot is not None:
        write_json(run_dir / "task.json", snapshot)
    trace = TraceWriter(run_dir / "trace.jsonl")
    trace.append(
        TraceEvent(
            run_id=run_id,
            kind="result.ingested",
            payload={
                "backend": result.backend,
                "trial_id": result.trial_id,
                "outcome": result.outcome,
                "source": result.source,
            },
        )
    )
    trace.close()
    return result, run_dir
