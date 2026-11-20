"""The model-first CLI surface, separate from legacy rollout and data commands."""

import argparse
import json
import sys
import subprocess
import tomllib
from pathlib import Path


def argv_array(value):
    try:
        result = json.loads(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a JSON argv array") from error
    if not isinstance(result, list) or not all(isinstance(x, str) for x in result):
        raise argparse.ArgumentTypeError("expected a JSON array of strings")
    return tuple(result)


def add_commands(commands):
    commands.add_parser("harnesses", help="List lazy Harness plugins and model protocols")
    commands.add_parser("suites", help="List packaged integration suites")
    run = commands.add_parser("run", help="Evaluate a model with a selected Harness; own serving when needed")
    reference = run.add_mutually_exclusive_group(required=True)
    reference.add_argument("--model", "-m")
    reference.add_argument("--model-config", type=Path)
    run.add_argument(
        "--harness", default="codex", help="codex (default), json, tools, harbor:<agent>, or installed plugin"
    )
    selection = run.add_mutually_exclusive_group()
    selection.add_argument("--suite", choices=("rua-smoke", "cua-smoke"))
    selection.add_argument("--tasks", nargs="+", type=Path)
    run.add_argument("--model-backend", choices=("auto", "endpoint", "vllm", "llama-cpp", "command"))
    run.add_argument("--base-url")
    run.add_argument("--served-model-name")
    run.add_argument("--api-key-env")
    run.add_argument("--model-revision")
    run.add_argument("--model-protocol", choices=("chat", "responses"))
    run.add_argument("--vision", action="store_true", default=None)
    run.add_argument("--server-executable")
    run.add_argument("--server-args", type=argv_array)
    run.add_argument("--server-command", type=argv_array)
    run.add_argument("--startup-timeout", type=float)
    run.add_argument("--runs-dir", type=Path, default=Path("runs"))
    run.add_argument("--batch-id")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--max-workers", type=int, default=1)
    run.add_argument("--cpu-budget", type=int)
    run.add_argument("--gpu-devices", default="")
    run.add_argument("--max-infra-retries", type=int, choices=(0, 1), default=1)
    run.add_argument("--model-retries", type=int, choices=range(6), default=2)
    run.add_argument("--context-images", type=int)
    run.add_argument("--context-mb", type=int, default=8)
    run.add_argument("--max-output-tokens", type=int, default=2048)
    run.add_argument("--max-total-tokens", type=int)
    run.add_argument("--temperature", type=float, default=0)
    run.add_argument("--top-p", type=float, default=1)
    run.add_argument("--sampling-seed", type=int)
    run.add_argument("--endpoint-concurrency", type=int, default=4)
    run.add_argument("--endpoint-rpm", type=int, default=0)
    run.add_argument("--model-request-timeout", type=float, default=120)
    run.add_argument("--harbor-executable", default="harbor")
    run.add_argument("--harbor-arg", action="append", default=[])
    run.add_argument("--codex-executable", default="codex")
    run.add_argument("--codex-bridge-host", help="Runner IP reachable from Docker; normally detected automatically")
    run.add_argument("--codex-decision-mode", choices=("json", "tools"), default="json")
    run.add_argument("--codex-context-window", type=int, default=32768)


def dispatch(args, parser):
    from worldline.harnesses import get_harness, harness_plugins
    from worldline.suites import builtin_suites

    if args.command == "harnesses":
        print(
            json.dumps(
                [
                    {"name": p.name, "version": p.version, "backend": p.backend, "protocols": p.protocols}
                    for p in harness_plugins().values()
                ],
                indent=2,
            )
        )
        return 0
    if args.command == "suites":
        print(json.dumps(builtin_suites(), indent=2))
        return 0
    from worldline.models.config import ModelConfig
    from worldline.evaluation import run_evaluation

    try:
        values = {}
        if args.model_config:
            config = tomllib.loads(args.model_config.read_text())
            if set(config) != {"model"} or not isinstance(config["model"], dict):
                raise ValueError("model config must contain exactly one [model] table")
            values = config["model"]
        overrides = {
            "reference": args.model,
            "backend": args.model_backend,
            "base_url": args.base_url,
            "name": args.served_model_name,
            "api_key_env": args.api_key_env,
            "revision": args.model_revision,
            "protocol": args.model_protocol,
            "vision": args.vision,
            "server_executable": args.server_executable,
            "server_args": args.server_args,
            "server_command": args.server_command,
            "startup_timeout_sec": args.startup_timeout,
        }
        values.update({key: value for key, value in overrides.items() if value is not None})
        if "protocol" not in values:
            values["protocol"] = get_harness(args.harness).protocols[0]
        model = ModelConfig(**values)
        options = {
            "max_retries": args.model_retries,
            "max_context_images": args.context_images,
            "max_context_bytes": args.context_mb * 1024 * 1024,
            "max_output_tokens": args.max_output_tokens,
            "max_total_tokens": args.max_total_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "sampling_seed": args.sampling_seed,
            "endpoint_concurrency": args.endpoint_concurrency,
            "endpoint_rpm": args.endpoint_rpm,
            "request_timeout_sec": args.model_request_timeout,
        }
        if args.harness == "codex":
            options["codex"] = {
                "executable": args.codex_executable,
                "bridge_host": args.codex_bridge_host,
                "decision_mode": args.codex_decision_mode,
                "context_window": args.codex_context_window,
            }
        print(
            f"Worldline: model={model.reference}, backend={model.effective_backend}, Harness={args.harness}",
            file=sys.stderr,
        )
        result = run_evaluation(
            model,
            harness=args.harness,
            suite=args.suite,
            tasks=args.tasks or (),
            runs_dir=args.runs_dir,
            batch_id=args.batch_id,
            resume=args.resume,
            dry_run=args.dry_run,
            max_workers=args.max_workers,
            agent_options=options,
            cpu_budget=args.cpu_budget,
            gpu_devices=tuple(d.strip() for d in args.gpu_devices.split(",") if d.strip()),
            max_infra_retries=args.max_infra_retries,
            harbor_executable=args.harbor_executable,
            harbor_args=args.harbor_arg,
        )
    except (ValueError, TypeError, OSError, subprocess.SubprocessError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in {"completed", "planned"} else 1
