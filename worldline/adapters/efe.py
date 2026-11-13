"""Explicit, optional EFE checkout boundary. Never installs import hooks or clones."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

REPOSITORY = "https://github.com/mll-lab-nu/embodied-first-exam.git"
PINNED_COMMIT = "e695c327deacb346fb468564a5cf40b72569cd34"
CHECKOUT_PATH = Path("third_party/embodied-first-exam")


def is_checkout(root: Path) -> bool:
    return all((root / path).exists() for path in ("tasks", "docs/core_set.toml", "scripts/task_conventions.py"))


def resolve_checkout(
    root: Path | None = None, *, start: Path | None = None, environ: Mapping[str, str] | None = None
) -> Path:
    environ = os.environ if environ is None else environ
    explicit = root or environ.get("WORLDLINE_EFE_ROOT") or environ.get("EFE_ROOT")
    if explicit:
        selected = Path(explicit).expanduser().resolve()
        if not is_checkout(selected):
            raise ValueError(f"not an EFE checkout: {selected}")
        return selected
    current = (start or Path.cwd()).resolve()
    for parent in (current, *current.parents):
        for candidate in (parent / CHECKOUT_PATH, parent):
            if is_checkout(candidate):
                return candidate
    raise ValueError(
        "EFE checkout unavailable: initialize third_party/embodied-first-exam with "
        "`git submodule update --init third_party/embodied-first-exam`, or set WORLDLINE_EFE_ROOT"
    )


def checkout_info(root: Path | None = None, *, start: Path | None = None) -> dict:
    path = resolve_checkout(root, start=start)
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        lines = result.stdout.splitlines()
        commit = lines[1] if result.returncode == 0 and Path(lines[0]).resolve() == path else None
    except (OSError, subprocess.TimeoutExpired):
        commit = None
    return {
        "repository": REPOSITORY,
        "root": str(path),
        "commit": commit,
        "pinned_commit": PINNED_COMMIT,
        "matches_pin": commit == PINNED_COMMIT,
    }


def launch(argv: Sequence[str], root: Path | None = None, *, python: str | None = None) -> int:
    path = resolve_checkout(root)
    env = {**os.environ, "EFE_ROOT": str(path)}
    # Run only the explicitly selected upstream CLI in a child process. Its own
    # optional dependencies are installed separately; Worldline imports none.
    return subprocess.call([python or sys.executable, str(path / "scripts/efe.py"), *argv], env=env)
