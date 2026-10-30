"""Host capability checks for Worldline and Harbor sandboxes."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _version(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = (completed.stdout or completed.stderr).strip().splitlines()
    return text[0] if text else None


def doctor(root: Path = Path.cwd()) -> dict[str, Any]:
    harbor = shutil.which("harbor")
    docker = shutil.which("docker")
    podman = shutil.which("podman")
    uv = shutil.which("uv")
    nvidia_smi = shutil.which("nvidia-smi")
    checks = {
        "python": {"ok": sys.version_info >= (3, 12), "version": sys.version.split()[0]},
        "uv": {"ok": uv is not None, "path": uv, "version": _version([uv, "--version"]) if uv else None},
        "harbor": {"ok": harbor is not None, "path": harbor, "version": _version([harbor, "--version"]) if harbor else None},
        "container": {
            "ok": docker is not None or podman is not None,
            "docker": docker,
            "podman": podman,
            "version": _version([docker, "--version"]) if docker else _version([podman, "--version"]) if podman else None,
        },
        "nvidia": {
            "ok": nvidia_smi is not None,
            "path": nvidia_smi,
            "version": _version([nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader"]) if nvidia_smi else None,
        },
        "repository": {
            "ok": (root / "tasks").is_dir() and (root / "worldline").is_dir(),
            "root": str(root.resolve()),
        },
    }
    recommendations: list[str] = []
    if not checks["harbor"]["ok"]:
        recommendations.append("Install the project with dependencies so the `harbor` executable is available.")
    if not checks["container"]["ok"]:
        recommendations.append("Install Docker Compose v2 or rootless Podman before running EFE sandboxes.")
    if not checks["nvidia"]["ok"]:
        recommendations.append("GPU tasks require an NVIDIA driver and the matching container runtime integration.")
    return {
        "schema_version": "worldline.doctor/0.1",
        "framework_ready": checks["python"]["ok"] and checks["uv"]["ok"] and checks["repository"]["ok"],
        "evaluation_ready": checks["harbor"]["ok"] and checks["container"]["ok"],
        "checks": checks,
        "recommendations": recommendations,
    }
