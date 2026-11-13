"""Independent core, native dependency and Harbor container health checks."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from worldline.adapters.efe import checkout_info


def _version(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = (completed.stdout or completed.stderr).strip().splitlines()
    return text[0] if text else None


def _probe(command: list[str]) -> dict:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
        return {"ok": result.returncode == 0, "detail": (result.stdout or result.stderr).strip()[:1000]}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"ok": False, "detail": str(error)}


def doctor(root: Path | None = None) -> dict[str, Any]:
    root = (root or Path.cwd()).resolve()
    harbor, docker, podman, uv, nvidia = (
        shutil.which(name) for name in ("harbor", "docker", "podman", "uv", "nvidia-smi")
    )
    engine = docker or podman
    container = _probe([engine, "info"]) if engine else {"ok": False, "detail": "no container engine installed"}
    compose = _probe([engine, "compose", "version"]) if engine else {"ok": False, "detail": "no compose installed"}
    gpu = (
        _probe([nvidia, "--query-gpu=index,name", "--format=csv,noheader"])
        if nvidia
        else {"ok": False, "detail": "no NVIDIA tools installed"}
    )
    gpu["ok"] = gpu["ok"] and bool(gpu.get("detail", "").strip())
    try:
        efe = {"ok": True, **checkout_info(start=root)}
    except ValueError as error:
        efe = {"ok": False, "detail": str(error)}
    checks = {
        "python": {"ok": (3, 12) <= sys.version_info[:2] < (3, 15), "version": sys.version.split()[0]},
        "uv": {"ok": uv is not None, "path": uv, "version": _version([uv, "--version"]) if uv else None},
        "harbor": {
            "ok": harbor is not None,
            "path": harbor,
            "version": _version([harbor, "--version"]) if harbor else None,
        },
        "container": {**container, "docker": docker, "podman": podman},
        "compose": compose,
        "nvidia": {**gpu, "path": nvidia},
        "repository": {"ok": (root / "worldline").is_dir(), "root": str(root)},
        "efe": efe,
    }
    rua_missing = [name for name in ("mujoco", "numpy", "PIL") if importlib.util.find_spec(name) is None]
    cua = {"dependencies_ready": False}
    try:
        from importlib.metadata import version
        from worldline.adapters.browsergym import BrowserGymEnvironment

        version("browsergym-core")
        BrowserGymEnvironment.preflight()
        cua = {"dependencies_ready": True}
    except Exception as error:
        cua["detail"] = str(error)
    recommendations = []
    if not checks["harbor"]["ok"]:
        recommendations.append("Harbor tasks: install worldline-agent[harbor] in the runner environment.")
    if not container["ok"] or not compose["ok"]:
        recommendations.append("Harbor tasks: make Docker/Podman daemon and Compose accessible to this user.")
    if not checks["nvidia"]["ok"]:
        recommendations.append("GPU tasks: install NVIDIA drivers and the container runtime integration.")
    return {
        "schema_version": "worldline.doctor/0.2",
        # Installed wheels work anywhere, including without a checkout, uv or EFE.
        "framework_ready": checks["python"]["ok"],
        "evaluation_ready": checks["python"]["ok"] and checks["harbor"]["ok"] and container["ok"] and compose["ok"],
        "native": {"rua": {"dependencies_ready": not rua_missing, "missing": rua_missing}, "cua": cua},
        "checks": checks,
        "recommendations": recommendations,
        "scope": "Dependency/container preflight only; native readiness requires an actual task rollout.",
    }
