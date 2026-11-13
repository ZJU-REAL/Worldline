"""Explicit native worker containment; trusted verifier stays runner-owned."""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from worldline.runtime.provisioning import state_root


def worker_command(definition, run_dir, python, worker_env=None):
    isolation = definition.task.metadata.get("isolation", {})
    backend = isolation.get("backend", "process")
    if backend == "process":
        return [python, "-m", "worldline.runtime.worker"], None
    if backend != "docker":
        raise ValueError("native isolation backend must be process or docker")
    image = isolation.get("image")
    import re

    if not isinstance(image, str) or not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", image):
        raise ValueError("container image must be pinned by sha256 digest")
    if not shutil.which("docker"):
        raise ValueError("docker isolation requires the Docker CLI and daemon")
    resources = definition.task.metadata.get("resources", {})
    cpus = resources.get("min_cpus", 1)
    memory_mb = isolation.get("memory_mb", 4096)
    pids = isolation.get("pids_limit", 256)
    if type(memory_mb) is not int or memory_mb < 64 or type(pids) is not int or pids < 1:
        raise ValueError("invalid container memory or process limit")
    name = "worldline-" + uuid.uuid4().hex
    source = Path(__file__).resolve().parents[2]
    state = state_root()
    state.mkdir(parents=True, exist_ok=True)
    command = [
        "docker",
        "run",
        "--rm",
        "--init",
        "--interactive",
        "--name",
        name,
        "--network",
        isolation.get("network", "none"),
        "--cpus",
        str(cpus),
        "--memory",
        f"{memory_mb}m",
        "--pids-limit",
        str(pids),
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--mount",
        f"type=bind,src={source},dst={source},readonly",
        "--mount",
        f"type=bind,src={run_dir.resolve()},dst={run_dir.resolve()}",
        "--env",
        f"PYTHONPATH={source}",
        "--workdir",
        str(run_dir.resolve()),
    ]
    command += ["--mount", f"type=bind,src={state},dst={state}", "--env", f"WORLDLINE_STATE_DIR={state}"]
    for asset in isolation.get("mounts", []):
        path = Path(asset).resolve(strict=True)
        command += ["--mount", f"type=bind,src={path},dst={path},readonly"]
    # Credentials are intentionally not inherited by the simulator container.
    if resources.get("gpu_required"):
        device = (worker_env or {}).get("WORLDLINE_GPU_DEVICE", os.environ.get("WORLDLINE_GPU_DEVICE"))
        if not device:
            raise ValueError("GPU container requires an admitted physical device")
        command += ["--gpus", f"device={device}"]
        command += ["--env", "WORLDLINE_GPU_DEVICE=0", "--env", f"WORLDLINE_HOST_GPU_DEVICE={device}"]
    command += [image, "python", "-m", "worldline.runtime.worker"]
    return command, name


def remove_container(name):
    if name:
        import subprocess

        subprocess.run(
            ["docker", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15, check=False
        )
