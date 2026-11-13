"""Worldline command-line entry point."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path
from typing import Sequence

from worldline import __version__
from worldline.adapters.harbor import load_task
from worldline.adapters.efe import checkout_info, launch as launch_efe
from worldline.batch import evaluate_many_harbor
from worldline.datasets import export_dataset, split_dataset, validate_datasets
from worldline.distill import build_distillation_bundle
from worldline.doctor import doctor
from worldline.eval import BackendUnavailableError, evaluate_harbor
from worldline.results import ingest_harbor_trial
from worldline.report import write_report
from worldline.adapters.rua.catalog import catalog_task_paths, catalog_task_metadata, write_catalog
from worldline.learning.launch import train
from worldline.learning.handoff import export_handoff
from worldline.diagnostics import profile_trace
from worldline.plugins import environment_plugins
from worldline.runtime.batch import evaluate_many_native
from worldline.runtime.queue import batch_status, request_stop
from worldline.tasks import load_native_task


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="worldline", description="Train, distill, and evaluate sandboxed agents")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    inspect_parser = commands.add_parser("inspect", help="Inspect a native or Harbor-compatible task")
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
    ingest_parser.add_argument("--task-dir", type=Path, help="Attach an explicitly reviewed source task snapshot")

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

    doctor_parser = commands.add_parser("doctor", help="Check core, native dependencies and Harbor separately")
    doctor_parser.add_argument("--backend", choices=("framework", "harbor", "rua", "cua"), default="framework")
    upstream = commands.add_parser("upstream-info", help="Inspect the explicitly selected EFE checkout")
    upstream.add_argument("--root", type=Path)
    efe = commands.add_parser("efe", help="Run the optional upstream EFE CLI in a child process")
    efe.add_argument("--root", type=Path)
    efe.add_argument("--python", help="Use a separate Python environment with EFE's optional dependencies")
    efe.add_argument("args", nargs=argparse.REMAINDER)

    distill_parser = commands.add_parser("distill", help="Build a complete training-data bundle from evaluation runs")
    distill_parser.add_argument("inputs", type=Path, nargs="+")
    distill_parser.add_argument("--output-dir", type=Path, required=True)
    distill_parser.add_argument("--train-ratio", type=float, default=0.8)
    distill_parser.add_argument("--validation-ratio", type=float, default=0.1)
    distill_parser.add_argument("--test-ratio", type=float, default=0.1)
    distill_parser.add_argument("--seed", default="worldline-v1")
    distill_parser.add_argument("--include-failures-in-sft", action="store_true")
    distill_parser.add_argument("--max-pairs-per-task", type=int, default=50)

    commands.add_parser("plugins", help="List available environment plugins without importing simulators")
    rollout = commands.add_parser("rollout", help="Evaluate native tasks through a registered environment")
    rollout.add_argument("task_dirs", nargs="+", type=Path)
    rollout.add_argument(
        "--agent", "-a", choices=("reference", "cua-reference", "noop", "openai-compatible"), required=True
    )
    rollout.add_argument("--model", "-m")
    rollout.add_argument("--base-url", default="http://localhost:8000/v1")
    rollout.add_argument("--api-key-env", default="WORLDLINE_API_KEY")
    rollout.add_argument("--runs-dir", type=Path, default=Path("runs"))
    rollout.add_argument("--max-workers", type=int, default=1)
    rollout.add_argument("--dry-run", action="store_true")
    rollout.add_argument("--batch-id")
    rollout.add_argument("--resume", action="store_true")
    rollout.add_argument("--max-infra-retries", type=int, choices=(0, 1), default=1)
    rollout.add_argument("--model-retries", type=int, default=2)
    rollout.add_argument("--harness-version", default="native-tools/0.2")
    rollout.add_argument("--cpu-budget", type=int)
    rollout.add_argument("--gpu-devices", default="")
    for name, help_text in (
        ("batch-status", "Read live durable batch status"),
        ("batch-stop", "Stop new trials and drain active ones"),
    ):
        control = commands.add_parser(name, help=help_text)
        control.add_argument("batch_id")
        control.add_argument("--runs-dir", type=Path, default=Path("runs"))

    handoff = commands.add_parser("training-handoff", help="Prepare a portable LLaMA-Factory or verl data package")
    handoff.add_argument("train_file", type=Path)
    handoff.add_argument("--validation-file", type=Path)
    handoff.add_argument("--framework", choices=("llama-factory", "verl"), required=True)
    handoff.add_argument("--output-dir", type=Path, required=True)
    handoff.add_argument("--model", default="REPLACE_WITH_MODEL")
    handoff.add_argument("--template", default="qwen")
    profile = commands.add_parser("profile-trace", help="Measure trace/media storage with a reproducible fixture")
    profile.add_argument("--output-dir", type=Path, required=True)
    profile.add_argument("--events", type=int, default=100)
    profile.add_argument("--unique-images", type=int, default=1)

    catalog = commands.add_parser("rua-catalog", help="Index existing RUA/EFE task packages without copying them")
    catalog.add_argument("root", type=Path)
    catalog.add_argument("--mode", choices=("privileged", "standard", "standard-realtime"))
    catalog.add_argument("--split")
    catalog.add_argument("--benchmark")
    catalog.add_argument("--output", "-o", type=Path, required=True)

    eval_catalog = commands.add_parser("eval-catalog", help="Evaluate an indexed Harbor task selection")
    eval_catalog.add_argument("catalog", type=Path)
    eval_catalog.add_argument("--agent", "-a", required=True)
    eval_catalog.add_argument("--model", "-m")
    eval_catalog.add_argument("--runs-dir", type=Path, default=Path("runs"))
    eval_catalog.add_argument("--max-workers", type=int, default=1)
    eval_catalog.add_argument("--harbor-executable", default="harbor")
    eval_catalog.add_argument("--dry-run", action="store_true")

    training = commands.add_parser("train", help="Validate inputs and run an independent training backend")
    training.add_argument("config", type=Path)
    training.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "inspect":
        task = (
            load_native_task(args.task_dir).task
            if (args.task_dir / "worldline.toml").is_file()
            else load_task(args.task_dir)
        )
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
    if args.command == "plugins":
        print(
            json.dumps(
                [{"name": p.name, "domain": p.domain, "version": p.version} for p in environment_plugins().values()],
                indent=2,
            )
        )
        return 0
    if args.command == "rollout":
        if args.agent == "openai-compatible":
            if not args.model:
                parser.error("openai-compatible requires --model")
            from worldline.agents.tool_agent import OpenAICompatibleAgent

            factory = partial(
                OpenAICompatibleAgent,
                model=args.model,
                base_url=args.base_url,
                api_key_env=args.api_key_env,
                max_retries=args.model_retries,
            )
        else:
            from worldline.domains.rua.reference import ReachReferenceAgent
            from worldline.domains.cua.reference import FormReferenceAgent

            def factory(**kwargs):
                policy = FormReferenceAgent if kwargs["spec"].get("domain") == "cua" else ReachReferenceAgent
                return policy(noop=args.agent == "noop", **kwargs)

        manifest = evaluate_many_native(
            args.task_dirs,
            agent_factory=factory,
            agent_name=args.agent,
            model=args.model,
            runs_dir=args.runs_dir,
            max_workers=args.max_workers,
            dry_run=args.dry_run,
            harness_version=args.harness_version,
            agent_config={
                "base_url": args.base_url,
                "api_key_env": args.api_key_env,
                "model_retries": args.model_retries,
            }
            if args.agent == "openai-compatible"
            else {"policy": args.agent},
            batch_id=args.batch_id,
            resume=args.resume,
            max_infra_retries=args.max_infra_retries,
            cpu_budget=args.cpu_budget,
            gpu_devices=tuple(device for device in args.gpu_devices.split(",") if device),
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 2 if any(manifest["statuses"].get(key) for key in ("error", "infra_error", "blocked", "pending")) else 0
    if args.command in {"batch-status", "batch-stop"}:
        try:
            result = (batch_status if args.command == "batch-status" else request_stop)(args.batch_id, args.runs_dir)
        except (ValueError, OSError) as error:
            parser.error(str(error))
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "training-handoff":
        try:
            manifest = export_handoff(
                args.train_file,
                args.output_dir,
                framework=args.framework,
                validation_file=args.validation_file,
                model=args.model,
                template=args.template,
            )
        except (ValueError, OSError, KeyError) as error:
            parser.error("\n".join([str(error), *getattr(error, "__notes__", [])]))
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    if args.command == "profile-trace":
        report = profile_trace(args.output_dir, events=args.events, unique_images=args.unique_images)
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "rua-catalog":
        data = write_catalog(args.root, args.output, mode=args.mode, split=args.split, benchmark=args.benchmark)
        print(json.dumps({"tasks": data["count"], "errors": data["errors"], "output": str(args.output)}, indent=2))
        return 1 if data["errors"] else 0
    if args.command == "eval-catalog":
        manifest = evaluate_many_harbor(
            catalog_task_paths(args.catalog),
            agent=args.agent,
            model=args.model,
            runs_dir=args.runs_dir,
            max_workers=args.max_workers,
            executable=args.harbor_executable,
            dry_run=args.dry_run,
            task_metadata=catalog_task_metadata(args.catalog),
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 2 if any(manifest["statuses"].get(key) for key in ("error", "infra_error")) else 0
    if args.command == "train":
        try:
            manifest, path = train(args.config, dry_run=args.dry_run)
        except (ValueError, OSError, KeyError) as error:
            parser.error(str(error))
        print(json.dumps({"status": manifest["status"], "manifest": str(path)}, indent=2))
        return manifest.get("exit_code", 0)
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
        result, run_dir = ingest_harbor_trial(args.trial_dir, args.runs_dir, task_dir=args.task_dir)
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
    if args.command == "doctor":
        diagnosis = doctor()
        print(json.dumps(diagnosis, ensure_ascii=False, indent=2))
        ready = (
            diagnosis["framework_ready"]
            if args.backend == "framework"
            else diagnosis["evaluation_ready"]
            if args.backend == "harbor"
            else diagnosis["native"][args.backend]["dependencies_ready"]
        )
        return 0 if ready else 1
    if args.command in {"upstream-info", "efe"}:
        try:
            if args.command == "upstream-info":
                print(json.dumps(checkout_info(args.root), indent=2))
                return 0
            return launch_efe(args.args[1:] if args.args[:1] == ["--"] else args.args, args.root, python=args.python)
        except (ValueError, OSError) as error:
            parser.error(str(error))
    if args.command == "distill":
        manifest = build_distillation_bundle(
            args.inputs,
            args.output_dir,
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            include_failures_in_sft=args.include_failures_in_sft,
            max_pairs_per_task=args.max_pairs_per_task,
        )
        print(
            json.dumps(
                {
                    "output_dir": manifest["output_dir"],
                    "report": manifest["report"],
                    "records": {name: data["export"]["records"] for name, data in manifest["datasets"].items()},
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
