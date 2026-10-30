"""Worldline command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from worldline import __version__
from worldline.adapters.harbor import load_task
from worldline.batch import evaluate_many_harbor
from worldline.datasets import export_dataset, split_dataset, validate_datasets
from worldline.eval import BackendUnavailableError, evaluate_harbor
from worldline.results import ingest_harbor_trial
from worldline.report import write_report


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

    ingest_parser = commands.add_parser("ingest", help="Normalize an existing Harbor trial")
    ingest_parser.add_argument("trial_dir", type=Path)
    ingest_parser.add_argument("--runs-dir", type=Path, default=Path("runs"))

    export_parser = commands.add_parser("export", help="Export verified trajectories for model training")
    export_parser.add_argument("inputs", type=Path, nargs="+")
    export_parser.add_argument("--format", choices=("sft", "preference", "rl"), required=True)
    export_parser.add_argument("--output", "-o", type=Path, required=True)
    export_parser.add_argument("--include-failures", action="store_true")
    export_parser.add_argument("--max-pairs-per-task", type=int, default=50)

    report_parser = commands.add_parser("report", help="Summarize normalized evaluation results")
    report_parser.add_argument("inputs", type=Path, nargs="+")
    report_parser.add_argument("--output", "-o", type=Path, required=True)

    batch_parser = commands.add_parser("eval-many", help="Evaluate multiple tasks with controlled concurrency")
    batch_parser.add_argument("task_dirs", type=Path, nargs="+")
    batch_parser.add_argument("--agent", "-a", required=True)
    batch_parser.add_argument("--model", "-m")
    batch_parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    batch_parser.add_argument("--harbor-executable", default="harbor")
    batch_parser.add_argument("--harbor-arg", action="append", default=[])
    batch_parser.add_argument("--max-workers", type=int, default=1)
    batch_parser.add_argument("--dry-run", action="store_true")

    split_parser = commands.add_parser("split", help="Split a dataset without leaking tasks across partitions")
    split_parser.add_argument("input", type=Path)
    split_parser.add_argument("--output-dir", type=Path, required=True)
    split_parser.add_argument("--train-ratio", type=float, default=0.8)
    split_parser.add_argument("--validation-ratio", type=float, default=0.1)
    split_parser.add_argument("--test-ratio", type=float, default=0.1)
    split_parser.add_argument("--seed", default="worldline-v1")

    validate_parser = commands.add_parser("validate-dataset", help="Validate JSONL data and task split isolation")
    validate_parser.add_argument("paths", type=Path, nargs="+")
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
    if args.command == "ingest":
        result, run_dir = ingest_harbor_trial(args.trial_dir, args.runs_dir)
        print(
            json.dumps(
                {
                    "trial_id": result.trial_id,
                    "outcome": result.outcome,
                    "reward": result.reward,
                    "run_dir": str(run_dir),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if result.outcome != "infra_error" else 2
    if args.command == "export":
        manifest = export_dataset(
            args.inputs,
            args.output,
            format=args.format,
            include_failures=args.include_failures,
            max_pairs_per_task=args.max_pairs_per_task,
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    if args.command == "report":
        report = write_report(args.inputs, args.output)
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
        return 0
    if args.command == "eval-many":
        manifest = evaluate_many_harbor(
            args.task_dirs,
            agent=args.agent,
            model=args.model,
            runs_dir=args.runs_dir,
            executable=args.harbor_executable,
            extra_args=args.harbor_arg,
            dry_run=args.dry_run,
            max_workers=args.max_workers,
        )
        print(json.dumps({key: manifest[key] for key in ("tasks", "statuses")}, ensure_ascii=False, indent=2))
        return 1 if manifest["statuses"].get("error") else 0
    if args.command == "split":
        manifest = split_dataset(
            args.input,
            args.output_dir,
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    if args.command == "validate-dataset":
        validation = validate_datasets(args.paths)
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        return 0 if validation["valid"] else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
