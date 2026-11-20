"""Build an independent scratch image containing only the installed static Codex CLI."""

import hashlib
import json
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

from worldline.storage import file_sha256


def native_binary(executable="codex"):
    command = shutil.which(executable)
    if not command:
        raise ValueError("Codex CLI is absent; install @openai/codex or pass --codex-executable")
    path = Path(command).resolve()
    if path.suffix == ".js":
        candidates = list(path.parents[1].glob("node_modules/@openai/codex-*/vendor/*/bin/codex"))
        candidates += list(path.parents[1].glob("vendor/*/bin/codex"))
        arch = "aarch64" if platform.machine() in {"aarch64", "arm64"} else "x86_64"
        path = next((p for p in candidates if arch in str(p)), None)
        if path is None:
            raise ValueError("cannot locate Codex's native Linux executable; pass its path with --codex-executable")
    if platform.system() != "Linux":
        raise ValueError("native Codex containment currently requires a Linux runner")
    result = subprocess.run([str(path), "--version"], capture_output=True, text=True, timeout=10, check=True)
    version = result.stdout.strip()
    if not version.startswith("codex-cli "):
        raise ValueError("unexpected Codex executable version")
    if tuple(int(v) for v in version.split()[1].split(".")[:2]) < (0, 162):
        raise ValueError("native Codex requires CLI 0.162+; Harbor/Codex uses its separate upstream version")
    return path, version


def prepare_image(executable="codex"):
    if not shutil.which("docker"):
        raise ValueError("native Codex requires a Docker daemon for agent isolation")
    subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=15)
    binary, version = native_binary(executable)
    digest = file_sha256(binary)
    dockerfile = 'FROM scratch\nCOPY codex /codex\nLABEL worldline.codex.contract="0.1"\n'
    identity = hashlib.sha256((digest + dockerfile).encode()).hexdigest()
    tag = "worldline/codex:" + identity[:24]
    inspect = subprocess.run(["docker", "image", "inspect", tag], capture_output=True, text=True, timeout=15)
    if inspect.returncode:
        with tempfile.TemporaryDirectory(prefix="worldline-codex-build-") as root:
            directory = Path(root)
            shutil.copy2(binary, directory / "codex")
            (directory / "Dockerfile").write_text(dockerfile)
            result = subprocess.run(
                ["docker", "build", "--network", "none", "-t", tag, str(directory)],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode:
                raise ValueError("Codex container build failed: " + result.stderr[-1000:])
        inspect = subprocess.run(
            ["docker", "image", "inspect", tag], capture_output=True, text=True, check=True, timeout=15
        )
    image = json.loads(inspect.stdout)[0]["Id"]
    # A static binary needs no OS image or network download. Validate that it
    # actually starts in scratch before any expensive model/environment startup.
    result = subprocess.run(
        ["docker", "run", "--rm", "--network", "none", image, "/codex", "--version"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    if result.returncode or result.stdout.strip() != version:
        raise ValueError("Codex cannot execute in scratch; use the static Linux binary from the official CLI package")
    return {"image": image, "codex_version": version, "binary_sha256": digest}
