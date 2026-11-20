"""Model -> Harness -> versioned tasks -> trusted evaluation report."""

import fcntl
import hashlib
import json
import os
import shutil
import time
import uuid
from functools import partial
from pathlib import Path
from urllib.parse import urlparse

from worldline import __version__
from worldline.adapters.harbor import load_task
from worldline.harnesses import get_harness
from worldline.models.serving import ModelService, check_server, probe_model
from worldline.plugins import get_environment_plugin
from worldline.report import write_report
from worldline.run import write_json
from worldline.runtime.queue import validate_batch_id
from worldline.suites import suite_tasks
from worldline.tasks import load_native_task
from worldline.storage import file_sha256


def _completed_resume(directory, runs_dir, batch_id):
    path = directory / "result.json"
    if not path.is_file():
        return None
    result = json.loads(path.read_text())
    if result.get("status") != "completed":
        return None
    manifest = json.loads((Path(runs_dir) / "batches" / batch_id / "manifest.json").read_text())
    for row in manifest["runs"]:
        if row["status"] != "completed":
            return None
        for attempt in row.get("attempts", []):
            run_path = Path(attempt["run_dir"])
            if not run_path.is_dir():
                raise ValueError("completed attempt artifacts missing; restore them before resume")
        if row.get("result_sha256"):
            artifact = Path(row["run_dir"]) / "result.normalized.json"
            if not artifact.is_file() or file_sha256(artifact) != row["result_sha256"]:
                raise ValueError("completed result artifact missing or changed; restore it before resume")
        for relative, digest in row.get("result_hashes", {}).items():
            artifact = Path(row["run_dir"]) / relative
            if not artifact.is_file() or file_sha256(artifact) != digest:
                raise ValueError("completed Harbor result artifact missing or changed; restore it before resume")
    report_path = Path(result["report"])
    if not report_path.is_file() or file_sha256(report_path) != result.get("report_sha256"):
        raise ValueError("completed report missing or changed; restore it before resume")
    return {**result, "resumed": True}


def run_evaluation(
    model_config,
    *,
    harness="json",
    suite=None,
    tasks=(),
    runs_dir=Path("runs"),
    batch_id=None,
    resume=False,
    dry_run=False,
    max_workers=1,
    agent_options=None,
    cpu_budget=None,
    gpu_devices=(),
    max_infra_retries=1,
    harbor_executable="harbor",
    harbor_args=(),
):
    plugin = get_harness(harness)
    if model_config.protocol not in plugin.protocols:
        raise ValueError(f"Harness {harness} requires {plugin.protocols}, received {model_config.protocol}")
    if type(max_workers) is not int or max_workers < 1:
        raise ValueError("max_workers must be a positive integer")
    if resume and not batch_id:
        raise ValueError("resume requires --batch-id")
    if suite and tasks:
        raise ValueError("choose --suite or --tasks")
    suite_metadata = None
    if suite or not tasks:
        tasks, suite_metadata = suite_tasks(suite or "rua-smoke")
        if suite_metadata["vision"] and not model_config.vision:
            raise ValueError("this suite requires an image-capable model; pass --vision")
    tasks = [Path(p).resolve() for p in tasks]
    snapshots = []
    for path in tasks:
        native = (path / "worldline.toml").is_file()
        if native != (plugin.backend == "native"):
            raise ValueError(
                f"Harness {harness} uses {plugin.backend} tasks, but {path} has the other task format. "
                "Use tools/json for native tasks; use codex/harbor:<agent> with --tasks <Harbor task>."
            )
        definition = load_native_task(path) if native else None
        task = definition.task if definition else load_task(path)
        if definition:
            env = get_environment_plugin(task.environment)
        snapshots.append(
            {
                "path": str(path),
                "id": task.id,
                "version": task.version,
                "environment": task.environment,
                "split": task.split,
                "seed": task.seed,
                "environment_version": env.version if definition else task.environment,
                "budgets": {
                    key: getattr(definition, key) for key in ("timeout_sec", "max_actions", "max_turns", "reset_budget")
                }
                if definition
                else task.metadata.get("evaluation", {}),
            }
        )
    if plugin.backend == "harbor":
        if model_config.effective_backend != "endpoint":
            raise ValueError(
                "Harbor coding Harness requires an existing service reachable from its containers; supply --base-url"
            )
        if urlparse(model_config.base_url).hostname in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Harbor containers cannot use host loopback; supply their reachable model service URL")
        if harness == "codex" and "/" in (model_config.name or model_config.reference):
            raise ValueError("Harbor 0.23 Codex needs a served alias without '/'; pass --served-model-name ALIAS")
        if not dry_run and not shutil.which(harbor_executable):
            raise ValueError(
                "Harbor executable unavailable; install worldline-agent[harbor] and container dependencies"
            )
    # Fingerprint the stable model identity, never the transient owned server port.
    settings = model_config.identity()
    options = dict(agent_options or {})
    plan = {
        "schema_version": "worldline.evaluation/0.1",
        "worldline_version": __version__,
        "model": settings,
        "harness": {"name": plugin.name, "version": plugin.version, "backend": plugin.backend},
        "suite": suite or ("rua-smoke" if suite_metadata else None),
        "suite_metadata": suite_metadata,
        "tasks": snapshots,
        "agent_options": options,
        "max_workers": max_workers,
        "dry_run": dry_run,
        "resources": {"cpu_budget": cpu_budget, "gpu_devices": list(gpu_devices)},
        "max_infra_retries": max_infra_retries,
        "harbor_executable": harbor_executable,
        "harbor_args": list(harbor_args),
    }
    signature = hashlib.sha256(json.dumps(plan, sort_keys=True, allow_nan=False).encode()).hexdigest()
    batch_id = batch_id or f"model-{uuid.uuid4().hex}"
    validate_batch_id(batch_id)
    directory = Path(runs_dir).resolve() / "evaluations" / batch_id
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("this model evaluation is already owned by another runner") from None
        plan_path = directory / "plan.json"
        if plan_path.exists():
            previous = json.loads(plan_path.read_text())
            if not resume or previous.get("signature") != signature:
                raise ValueError(
                    "evaluation exists or configuration changed; use matching --resume or a new --batch-id"
                )
        elif resume:
            raise ValueError("evaluation not found; omit --resume for a new run")
        if resume and not dry_run and (completed := _completed_resume(directory, runs_dir, batch_id)):
            return completed
        write_json(plan_path, {**plan, "signature": signature, "batch_id": batch_id})
        if dry_run:
            result = {
                "status": "planned",
                "batch_id": batch_id,
                "plan": str(plan_path),
                "model": settings,
                "harness": plugin.name,
                "tasks": snapshots,
            }
            write_json(directory / "result.json", result)
            return result
        status = {"status": "starting", "batch_id": batch_id, "plan": str(plan_path)}
        write_json(directory / "result.json", status)
        try:
            if plugin.backend == "native":
                check_server(model_config)
                checked = set()
                for task_path in tasks:
                    env = get_environment_plugin(load_native_task(task_path).task.environment)
                    if env.preflight and env.name not in checked:
                        env.preflight()
                        checked.add(env.name)
            with ModelService(model_config, directory / ("service-" + uuid.uuid4().hex), settings=settings) as service:
                started = time.monotonic()
                probe = probe_model(service.model, tools=harness == "tools")
                write_json(
                    directory / "preflight.json",
                    {"readiness": service.readiness, "probe": probe, "duration_sec": time.monotonic() - started},
                )
                if plugin.backend == "native":
                    from worldline.runtime.batch import evaluate_many_native

                    factory = partial(plugin.factory, model=service.model, options=options)
                    manifest = evaluate_many_native(
                        tasks,
                        agent_factory=factory,
                        agent_name=plugin.name,
                        model=service.model.name,
                        harness_version=plugin.version,
                        agent_config={"model_source": settings, "options": options},
                        runs_dir=Path(runs_dir),
                        batch_id=batch_id,
                        resume=resume and (Path(runs_dir) / "batches" / batch_id / "state.sqlite").exists(),
                        max_workers=max_workers,
                        cpu_budget=cpu_budget,
                        gpu_devices=gpu_devices,
                        max_infra_retries=max_infra_retries,
                    )
                else:
                    from worldline.batch import evaluate_many_harbor

                    extra = list(harbor_args)
                    if harness == "codex":
                        # Native Codex configuration is carried by the pinned Harbor adapter.
                        extra += [
                            "--agent-kwarg",
                            "config_source=" + json.dumps({"openai_base_url": service.model.base_url}),
                        ]
                    env = {
                        "OPENAI_BASE_URL": service.model.base_url,
                        "OPENAI_API_KEY": os.environ.get(model_config.api_key_env) or "worldline-local",
                    }
                    manifest = evaluate_many_harbor(
                        tasks,
                        agent=plugin.harbor_agent,
                        model="openai/" + service.model.name if harness == "codex" else service.model.name,
                        runs_dir=Path(runs_dir),
                        batch_id=batch_id,
                        resume=resume and (Path(runs_dir) / "batches" / batch_id / "state.sqlite").exists(),
                        max_workers=max_workers,
                        executable=harbor_executable,
                        extra_args=extra,
                        env=env,
                        task_metadata={p: {"model_source": settings, "harness_version": plugin.version} for p in tasks},
                    )
            inputs = []
            for row in manifest["runs"]:
                inputs.extend(Path(a["run_dir"]) for a in row.get("attempts", []) if a.get("run_dir"))
                if row.get("run_dir"):
                    inputs.append(Path(row["run_dir"]))
            report_path = directory / "report.json"
            report = write_report(inputs, report_path)
            incomplete = report["summary"]["counted_trials"] < len(tasks) or any(
                row["status"] not in {"completed"} for row in manifest["runs"]
            )
            status = {
                "status": "incomplete" if incomplete or report["summary"]["outcomes"]["infra_error"] else "completed",
                "batch_id": batch_id,
                "model": model_config.reference,
                "harness": plugin.name,
                "plan": str(plan_path),
                "report": str(report_path),
                "report_sha256": file_sha256(report_path),
                "summary": report["summary"],
                "batch_statuses": manifest["statuses"],
            }
            write_json(directory / "result.json", status)
            return status
        except BaseException as error:
            write_json(directory / "result.json", {**status, "status": "failed", "error": str(error)})
            raise
