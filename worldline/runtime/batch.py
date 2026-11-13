"""Durable bounded rollouts, resume, and one fresh-environment infrastructure retry."""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from worldline import __version__
from worldline.runtime.resources import ResourcePool
from worldline.runtime.shared_resources import HostResourcePool
from worldline.runtime.state import DurableQueue
from worldline.runtime.queue import snapshot_batch, validate_batch_id
from worldline.plugins import get_environment_plugin
from worldline.run import write_json
from worldline.storage import file_sha256
from worldline.runtime.session import evaluate_native
from worldline.tasks import load_native_task


def evaluate_many_native(
    task_dirs,
    *,
    max_workers=1,
    runs_dir=Path("runs"),
    batch_id=None,
    resume=False,
    max_infra_retries=1,
    retry_delay_sec=0.5,
    cpu_budget=None,
    gpu_devices=(),
    resource_scope="host",
    resource_dir=None,
    coordinator_url=None,
    **options,
):
    if type(max_workers) is not int or max_workers < 1:
        raise ValueError("max_workers must be a positive integer")
    if max_infra_retries not in {0, 1} or retry_delay_sec < 0:
        raise ValueError("at most one fresh-environment infrastructure retry is supported")
    if resume and not batch_id:
        raise ValueError("resume requires an explicit batch_id")
    batch_id = batch_id or f"native-{uuid.uuid4().hex}"
    validate_batch_id(batch_id)
    directory = runs_dir.resolve() / "batches" / batch_id
    tasks = [Path(path).resolve() for path in task_dirs]
    identities = []
    for path in tasks:
        try:
            identities.append((str(path), load_native_task(path).task.version))
        except (OSError, ValueError, KeyError):
            identities.append((str(path), "invalid"))
    configuration = {
        key: options.get(key) for key in ("agent_name", "model", "harness_version", "dry_run", "agent_config")
    }
    from worldline.agents.config import factory_configuration

    configuration["agent_factory"] = factory_configuration(options.get("agent_factory"))
    occurrences, logical_jobs = {}, []
    for path in tasks:
        try:
            task_id = load_native_task(path).task.id
        except (OSError, ValueError, KeyError):
            task_id = str(path)
        count = occurrences.get(task_id, 0)
        occurrences[task_id] = count + 1
        logical_jobs.append(f"{batch_id}:{task_id}:{count}")
    signature = hashlib.sha256(
        json.dumps(
            {
                "tasks": identities,
                "config": configuration,
                "max_infra_retries": max_infra_retries,
                "worldline_version": __version__,
                "cpu_budget": cpu_budget,
                "gpu_devices": tuple(gpu_devices),
                "resource_scope": resource_scope,
                "resource_dir": str(resource_dir) if resource_dir else None,
                "coordinator_url": coordinator_url,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    state = DurableQueue(directory)
    db, mutex = state.db, state.mutex
    try:
        if state.initialize(tasks, signature, resume):
            for status, saved in db.execute(
                "SELECT status, result FROM jobs WHERE status IN ('completed', 'infra_error', 'planned')"
            ):
                row = json.loads(saved)
                if status == "planned":
                    continue
                final_path = Path(row["run_dir"]) / "result.normalized.json"
                try:
                    final = json.loads(final_path.read_text())
                    valid = final.get("trial_id") == row["run_id"] and final.get("outcome") in {
                        "success",
                        "failure",
                        "infra_error",
                    }
                    if row.get("result_sha256"):
                        valid = valid and file_sha256(final_path) == row["result_sha256"]
                except (OSError, ValueError):
                    valid = False
                if not valid:
                    raise ValueError(
                        f"completed batch artifact missing or invalid: {final_path}; restore artifacts before resume"
                    )
            db.execute("UPDATE jobs SET status='pending' WHERE status IN ('running', 'blocked')")
        state.reset_stop()
        capacity = cpu_budget if cpu_budget is not None else os.cpu_count() or 1
        if resource_scope == "host":
            resources = HostResourcePool(capacity, gpu_devices, directory=resource_dir)
        elif resource_scope == "process":
            resources = ResourcePool(capacity, gpu_devices)
        else:
            raise ValueError("resource_scope must be host or process")
        if resources.capacity < 1:
            raise ValueError("cpu_budget must be positive")
        provider_blocked = threading.Event()
        preflights = {}

        claim, save = state.claim, state.save

        def worker():
            while job := claim():
                ordinal, raw_path, previous_result = job
                path = Path(raw_path)
                row = json.loads(previous_result) if previous_result else {"task_dir": str(path), "attempts": []}
                if provider_blocked.is_set():
                    row.update(
                        status="blocked",
                        error="permanent model authentication/endpoint error; repair provider before resume",
                    )
                    save(ordinal, row, "blocked")
                    continue
                cpus, device, leased = 0, None, False
                remote = None
                remote_granted = False
                try:
                    definition = load_native_task(path)
                    plugin = get_environment_plugin(definition.task.environment)
                    if coordinator_url and not options.get("dry_run"):
                        from worldline.runtime.coordinator import RemoteLease

                        remote_signature = hashlib.sha256(
                            json.dumps(
                                {
                                    "task_version": definition.task.version,
                                    "config": configuration,
                                    "worldline_version": __version__,
                                },
                                sort_keys=True,
                            ).encode()
                        ).hexdigest()
                        remote = RemoteLease(coordinator_url, logical_jobs[ordinal], remote_signature)
                        remote_state = remote.claim()
                        remote_granted = remote_state["granted"]
                        if not remote_state["granted"]:
                            row.update(
                                status="remote_completed"
                                if remote_state["status"] == "completed"
                                else "orphaned"
                                if remote_state["status"] == "orphaned"
                                else "blocked",
                                remote_result=remote_state.get("result"),
                                error="job is owned/finalized by another runner; collect its original artifacts",
                            )
                            save(ordinal, row, row["status"])
                            continue
                    if plugin.preflight and not options.get("dry_run"):
                        key = (plugin.name, plugin.version)
                        with mutex:
                            if key not in preflights:
                                try:
                                    plugin.preflight()
                                    preflights[key] = None
                                except Exception as failure:
                                    preflights[key] = f"{type(failure).__name__}: {failure}"
                            failure = preflights[key]
                        if failure:
                            raise ValueError(f"environment preflight failed: {failure}")
                    if row["attempts"]:
                        final_path = Path(row["attempts"][-1]["run_dir"]) / "result.normalized.json"
                        if final_path.is_file():
                            final = json.loads(final_path.read_text())
                            if (
                                final.get("outcome") != "infra_error"
                                or not final.get("retryable")
                                or not plugin.restartable
                                or len(row["attempts"]) >= max_infra_retries + 1
                            ):
                                for previous_attempt in row["attempts"][:-1]:
                                    previous_path = Path(previous_attempt["run_dir"]) / "result.normalized.json"
                                    if previous_path.is_file():
                                        old = json.loads(previous_path.read_text())
                                        old.update(counted=False, superseded_by=final["trial_id"])
                                        write_json(previous_path, old)
                                row.update(
                                    status="infra_error" if final["outcome"] == "infra_error" else "completed",
                                    run_id=final["trial_id"],
                                    run_dir=str(final_path.parent),
                                    result_sha256=file_sha256(final_path),
                                )
                                save(ordinal, row, row["status"])
                                continue
                    requested = {} if options.get("dry_run") else definition.task.metadata.get("resources", {})
                    cpus = max(1, math.ceil(requested.get("min_cpus", 1)))
                    gpu = bool(requested.get("gpu_required", False))

                    def stopped():
                        with mutex:
                            value = db.execute("SELECT value FROM control WHERE key='stop_requested'").fetchone()
                        return (
                            provider_blocked.is_set()
                            or bool(value and value[0] == "1")
                            or bool(remote and remote.error)
                        )

                    device = resources.acquire(cpus, gpu, cancel=stopped)
                    leased = True
                    environment = (
                        {"CUDA_VISIBLE_DEVICES": str(device), "WORLDLINE_GPU_DEVICE": str(device)}
                        if device is not None
                        else None
                    )
                    restartable = get_environment_plugin(definition.task.environment).restartable
                    if row["attempts"] and not restartable:
                        raise ValueError("non-restartable environment cannot resume an interrupted trial")
                    # An interrupted attempt is preserved but never silently reused.
                    prior = row["attempts"][-1]["run_dir"] if row["attempts"] else None
                    remaining_attempts = max_infra_retries + 1 - len(row["attempts"])
                    if remaining_attempts < 1:
                        raise ValueError("batch attempt budget exhausted; inspect preserved attempts")
                    for retry in range(remaining_attempts):

                        def created(record, run_dir):
                            row["attempts"].append({"run_id": record.run_id, "run_dir": str(run_dir)})
                            save(ordinal, row)

                        record, run_dir = evaluate_native(
                            path,
                            runs_dir=runs_dir,
                            on_created=created,
                            logical_trial_id=f"{batch_id}:{ordinal}",
                            attempt=len(row["attempts"]) + 1,
                            worker_env=environment,
                            cancel_check=(lambda: bool(remote.error)) if remote else None,
                            **options,
                        )
                        row.update(run_id=record.run_id, run_dir=str(run_dir), status=record.status)
                        result_path = run_dir / "result.normalized.json"
                        result = json.loads(result_path.read_text()) if result_path.is_file() else {}
                        if result_path.is_file():
                            row["result_sha256"] = file_sha256(result_path)
                        if options.get("agent_name") == "openai-compatible" and result.get("error_code") in {
                            "model.http.401",
                            "model.http.403",
                            "model.http.404",
                        }:
                            provider_blocked.set()
                        if prior:
                            previous_path = Path(prior) / "result.normalized.json"
                            if previous_path.is_file():
                                old = json.loads(previous_path.read_text())
                                old.update(counted=False, superseded_by=record.run_id)
                                write_json(previous_path, old)
                        save(ordinal, row)
                        if remote and remote.error:
                            break
                        if not (
                            result.get("outcome") == "infra_error"
                            and result.get("retryable")
                            and restartable
                            and retry + 1 < remaining_attempts
                        ):
                            break
                        prior = str(run_dir)
                        time.sleep(retry_delay_sec)
                except InterruptedError:
                    row.update(status="pending")
                except Exception as error:
                    row.update(status="error", error=f"{type(error).__name__}: {error}")
                finally:
                    if remote:
                        try:
                            if remote_granted and row.get("status") != "pending":
                                remote.complete(
                                    {
                                        "run_id": row.get("run_id"),
                                        "run_dir": row.get("run_dir"),
                                        "result_sha256": row.get("result_sha256"),
                                        "status": row["status"],
                                    }
                                )
                        except Exception as error:
                            row.update(
                                status="orphaned",
                                error=f"distributed completion could not be confirmed: {type(error).__name__}",
                            )
                        finally:
                            remote.close()
                    if leased:
                        resources.release(cpus, device)
                    save(ordinal, row, row.get("status", "error"))

        # Only max_workers tasks are in flight; no future for every input task.
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(worker) for _ in range(max_workers)]
            try:
                for future in futures:
                    future.result()
            except KeyboardInterrupt:
                state.stop()
                # Active trials drain within their own budgets. No new trials
                # are claimed; pending jobs can resume without consuming retries.
                raise
        rows = snapshot_batch(directory / "state.sqlite")["runs"]
        statuses = {}
        for row in rows:
            statuses[row["status"]] = statuses.get(row["status"], 0) + 1
        manifest = {
            "schema_version": "worldline.batch/0.3",
            "worldline_version": __version__,
            "drained": any(row["status"] == "pending" for row in rows),
            "batch_id": batch_id,
            "backend": "native",
            "tasks": len(rows),
            "max_workers": max_workers,
            "statuses": statuses,
            "runs": rows,
            "resume": resume,
            "max_infra_retries": max_infra_retries,
            "resources": {"cpu_budget": resources.capacity, "gpu_devices": list(gpu_devices), "scope": resource_scope},
            "state": str(directory / "state.sqlite"),
            "coordinator_url": coordinator_url,
        }
        write_json(directory / "manifest.json", manifest)
        return manifest
    finally:
        state.close()
