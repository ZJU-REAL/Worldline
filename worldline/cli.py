"""Worldline command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from worldline import __version__
from worldline.adapters.harbor import load_task
from worldline.eval import BackendUnavailableError, evaluate_harbor


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="worldline", description="Train, distill, and evaluate sandboxed agents")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    inspect_parser = commands.add_parser("inspect", help="Inspect a Harbor-compatible task")
    inspect_parser.add_argument("task_dir", type=Path)

    eval_parser = commands.add_parser("eval", help="Evaluate one task and create a Worldline run record")
    eval_parser.add_argument("task_dir", type=Path)
    eval_parser.add_argument("--agent", "-a", required=True)
    eval_parser.add_argument("--model", "-m")
    eval_parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    eval_parser.add_argument("--harbor-executable", default="harbor")
    eval_parser.add_argument("--harbor-arg", action="append", default=[])
    eval_parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
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
    if args.command == "eval":
        try:
            record, run_dir = evaluate_harbor(
                args.task_dir,
                agent=args.agent,
                model=args.model,
                runs_dir=args.runs_dir,
                executable=args.harbor_executable,
                extra_args=args.harbor_arg,
                dry_run=args.dry_run,
            )
        except BackendUnavailableError as error:
            parser.error(str(error))
        print(
            json.dumps(
                {"run_id": record.run_id, "status": record.status, "run_dir": str(run_dir)},
                ensure_ascii=False,
                indent=2,
            )
        )
        return record.exit_code or 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
