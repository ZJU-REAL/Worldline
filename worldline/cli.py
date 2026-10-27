"""Worldline command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from worldline import __version__
from worldline.adapters.harbor import load_task


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="worldline", description="Train, distill, and evaluate sandboxed agents")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    inspect_parser = commands.add_parser("inspect", help="Inspect a Harbor-compatible task")
    inspect_parser.add_argument("task_dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "inspect":
        task = load_task(args.task_dir)
        print(
            json.dumps(
                {
                    "id": task.id,
                    "environment": task.environment,
                    "version": task.version,
                    "split": task.split,
                    "seed": task.seed,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
