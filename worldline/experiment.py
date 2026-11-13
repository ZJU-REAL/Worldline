"""Versioned evaluation/teacher/data plans; external trainers remain independent."""

from __future__ import annotations

import hashlib
import json
import tomllib
import fcntl
import uuid
from functools import partial
from pathlib import Path

from worldline.agents.tool_agent import OpenAICompatibleAgent
from worldline.batch import evaluate_many_harbor
from worldline.distill import build_distillation_bundle
from worldline.learning.handoff import export_handoff
from worldline.report import write_report
from worldline.runtime.batch import evaluate_many_native
from worldline.run import write_json
from worldline.storage import file_sha256, verify_package


def run_experiment(config_path, *, dry_run=False, resume=False):
    path = Path(config_path).resolve()
    raw = path.read_bytes()
    config = tomllib.loads(raw.decode())
    if config.get("schema_version") != "worldline.experiment/0.1":
        raise ValueError("unsupported experiment plan")
    root = (path.parent / config["output_dir"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    invocation = uuid.uuid4().hex
    with (root / "owner.lock").open("a") as owner:
        try:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("experiment is owned by another runner") from None
        try:
            return _run_experiment((path, raw, config), invocation, dry_run=dry_run, resume=resume)
        except BaseException as error:
            manifest_path = root / ("plan.json" if dry_run else "experiment.json")
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                if manifest.get("invocation_id") == invocation:
                    manifest.update(
                        passed=False,
                        status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        error=f"{type(error).__name__}: {error}",
                        notes=getattr(error, "__notes__", []),
                    )
                    write_json(manifest_path, manifest)
            raise


def _run_experiment(loaded, invocation, *, dry_run, resume):
    path, raw, config = loaded
    if config.get("schema_version") != "worldline.experiment/0.1":
        raise ValueError("unsupported experiment plan")
    root = (path.parent / config["output_dir"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    plan_hash = hashlib.sha256(raw).hexdigest()
    manifest_path = root / ("plan.json" if dry_run else "experiment.json")
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["plan_sha256"] != plan_hash or not (resume or dry_run):
            raise ValueError("experiment exists; resume requires an identical plan")
    phases = config.get("phase", [])
    if not phases or len({phase["id"] for phase in phases}) != len(phases):
        raise ValueError("experiment requires unique phase IDs")
    from worldline.runtime.queue import validate_batch_id

    for phase in phases:
        validate_batch_id(phase["id"])
    manifest = {
        "schema_version": "worldline.experiment_result/0.1",
        "plan_sha256": plan_hash,
        "dry_run": dry_run,
        "phases": [],
        "passed": True,
        "invocation_id": invocation,
        "status": "running",
    }
    write_json(manifest_path, manifest)
    for phase in phases:
        manifest.update(active_phase=phase["id"], stage="evaluation")
        write_json(manifest_path, manifest)
        tasks = [(path.parent / value).resolve() for value in phase["tasks"]]
        backend = phase.get("backend", "native")
        options = phase.get("model", {})
        runs = root / (".planned" if dry_run else "executions") / phase["id"] / "runs"
        phase_resume = (resume or dry_run) and (runs / "batches" / phase["id"] / "state.sqlite").exists()
        if backend == "native":
            factory = partial(OpenAICompatibleAgent, **options)
            result = evaluate_many_native(
                tasks,
                agent_factory=factory,
                agent_name="openai-compatible",
                model=options["model"],
                agent_config=options,
                runs_dir=runs,
                batch_id=phase["id"],
                dry_run=dry_run,
                resume=phase_resume,
                max_workers=phase.get("max_workers", 1),
                cpu_budget=phase.get("cpu_budget"),
                gpu_devices=tuple(phase.get("gpu_devices", ())),
            )
        elif backend == "harbor":
            result = evaluate_many_harbor(
                tasks,
                agent=phase["agent"],
                model=options.get("model"),
                runs_dir=runs,
                batch_id=phase["id"],
                dry_run=dry_run,
                resume=phase_resume,
                max_workers=phase.get("max_workers", 1),
            )
        else:
            raise ValueError("phase backend must be native or harbor")
        entry = {"id": phase["id"], "batch": result}
        failures = {
            key: value for key, value in result["statuses"].items() if key not in {"completed", "planned"} and value
        }
        manifest["passed"] = manifest["passed"] and not failures
        if not dry_run:
            report = root / phase["id"] / "report.json"
            write_report([runs], report)
            entry["report_sha256"] = file_sha256(report)
            if phase.get("distill") and not failures:
                manifest["stage"] = "distillation"
                write_json(manifest_path, manifest)
                data = root / phase["id"] / "dataset"
                from worldline.learning.quality import QualityPolicy

                entry["distillation"] = (
                    verify_package(data)
                    if (data / "manifest.json").exists()
                    else build_distillation_bundle(
                        [runs], data, quality_policy=QualityPolicy(**phase.get("quality", {}))
                    )
                )
                for framework in phase.get("handoff", []):
                    manifest["stage"] = framework + "-handoff"
                    write_json(manifest_path, manifest)
                    output = root / phase["id"] / framework
                    entry[framework] = (
                        verify_package(output)
                        if (output / "manifest.json").exists()
                        else export_handoff(
                            data / "splits/sft/train.jsonl",
                            output,
                            framework=framework,
                            model=phase.get("student_model", "REPLACE_WITH_MODEL"),
                            template=phase.get("template", "qwen"),
                            tool_prose_policy=phase.get("tool_prose_policy", "preserve"),
                        )
                    )
        manifest["phases"].append(entry)
        write_json(manifest_path, manifest)
        if failures:
            break
    manifest.update(
        status=("planned" if dry_run else "completed") if manifest["passed"] else "failed", stage="finished"
    )
    write_json(manifest_path, manifest)
    return manifest
