"""Repository checks and the requested human-only commit attribution policy."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from worldline.run import write_json


def check_commit_message(path: Path):
    bot = re.compile(
        r"^(?:claude(?: code| (?:opus|sonnet|haiku)(?:[ -]\d+(?:\.\d+)?)?)?|codex|github copilot)(?:\[bot\])?\s*<", re.I
    )
    for field in ("GIT_AUTHOR_IDENT", "GIT_COMMITTER_IDENT"):
        identity = subprocess.run(["git", "var", field], check=True, capture_output=True, text=True).stdout
        if bot.match(identity.strip()):
            raise ValueError("Use the repository's configured human commit identity; AI attribution is disabled.")
    for line in path.read_text().splitlines():
        if line.lower().startswith("co-authored-by:") and bot.match(line.split(":", 1)[1].strip()):
            raise ValueError("AI Co-authored-by trailers are disabled by this repository's attribution policy.")


def check_fingerprint(root: Path):
    digest = hashlib.sha256()
    for area in ("worldline", "tests", "examples", "docs/worldline"):
        for path in sorted((root / area).rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix in {".pyc", ".tmp"}:
                continue
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    for name in ("pyproject.toml", "README.md", "AGENTS.md", "CONTRIBUTING.md"):
        digest.update((root / name).read_bytes())
    packages = sorted((entry.metadata["Name"], entry.version) for entry in importlib.metadata.distributions())
    settings = {
        key: os.environ.get(key)
        for key in ("PLAYWRIGHT_BROWSERS_PATH", "WORLDLINE_CHROMIUM_PATH", "MUJOCO_GL", "CUDA_VISIBLE_DEVICES")
    }
    digest.update(
        json.dumps((sys.version, str(Path(sys.executable).resolve()), packages, settings), sort_keys=True).encode()
    )
    return digest.hexdigest()


def check_repository(root: Path, *, cached=False):
    root = root.resolve()
    report_dir = root / "reports/platform-check"
    stamp = report_dir / "success.json"
    signature = check_fingerprint(root)
    if cached and stamp.is_file():
        prior = json.loads(stamp.read_text())
        if prior.get("fingerprint") == signature and time.time() - prior.get("time", 0) < 900:
            print("Worldline checks already passed for this source and environment.")
            return 0
    report_dir.mkdir(parents=True, exist_ok=True)
    commands = (
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        [sys.executable, "-m", "ruff", "check", "worldline", "tests"],
        [sys.executable, "-m", "ruff", "format", "--check", "worldline", "tests"],
    )
    with (report_dir / "check.log").open("wb") as log:
        for command in commands:
            result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=False)
            if result.returncode:
                print(f"Checks failed; see {report_dir / 'check.log'}", file=sys.stderr)
                return result.returncode
    write_json(stamp, {"fingerprint": signature, "time": time.time(), "commands": commands})
    print(f"Worldline tests and Ruff passed; log: {report_dir / 'check.log'}")
    return 0


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    message = commands.add_parser("commit-msg")
    message.add_argument("path", type=Path)
    check = commands.add_parser("check")
    check.add_argument("--cached", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "commit-msg":
            check_commit_message(args.path)
            return 0
        return check_repository(Path.cwd(), cached=args.cached)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
