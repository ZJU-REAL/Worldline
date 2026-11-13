"""Build a reproducible training bundle from normalized evaluation runs."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Sequence

from worldline.datasets import export_datasets, split_dataset, validate_datasets
from worldline.report import write_report
from worldline.run import write_json
from worldline.storage import staged_directory, file_sha256


def _build_distillation_bundle(
    inputs: Sequence[Path],
    output_dir: Path,
    *,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: str = "worldline-v1",
    include_failures_in_sft: bool = False,
    max_pairs_per_task: int = 50,
    media_mode: str = "reference",
    allow_unspecified_split: bool = False,
    quality_policy=None,
) -> dict[str, Any]:
    """Export, split, and validate all supported training views."""

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = write_report(inputs, output_dir / "report.json")
    datasets: dict[str, Any] = {}

    exports = export_datasets(
        inputs,
        {name: output_dir / "raw" / f"{name}.jsonl" for name in ("sft", "preference", "rl")},
        include_failures=include_failures_in_sft,
        max_pairs_per_task=max_pairs_per_task,
        media_mode=media_mode,
        media_root=output_dir,
        allow_unspecified_split=allow_unspecified_split,
        quality_policy=quality_policy,
    )
    for format_name in ("sft", "preference", "rl"):
        raw_path = output_dir / "raw" / f"{format_name}.jsonl"
        export = exports[format_name]
        split_dir = output_dir / "splits" / format_name
        split = split_dataset(
            raw_path,
            split_dir,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
        split_paths = [split_dir / f"{name}.jsonl" for name in ("train", "validation", "test")]
        validation = validate_datasets(split_paths)
        if not validation["valid"]:
            raise ValueError(f"invalid {format_name} dataset: {validation['errors']}")
        datasets[format_name] = {
            "raw": str(raw_path),
            "export": export,
            "split": split,
            "validation": validation,
        }

    manifest = {
        "schema_version": "worldline.distillation_bundle/0.1",
        "inputs": [str(path.resolve()) for path in inputs],
        "output_dir": str(output_dir),
        "report": {"path": str(output_dir / "report.json"), "summary": report["summary"]},
        "split": {
            "seed": seed,
            "ratios": {"train": train_ratio, "validation": validation_ratio, "test": test_ratio},
        },
        "datasets": datasets,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def build_distillation_bundle(inputs: Sequence[Path], output_dir: Path, **options) -> dict[str, Any]:
    destination = output_dir.resolve()
    with staged_directory(destination) as stage:
        manifest = _build_distillation_bundle(inputs, stage, **options)

        def relocate(value):
            if isinstance(value, str) and (value == str(stage) or value.startswith(str(stage) + "/")):
                return str(destination) + value[len(str(stage)) :]
            if isinstance(value, dict):
                return {relocate(key): relocate(nested) for key, nested in value.items()}
            if isinstance(value, list):
                return [relocate(nested) for nested in value]
            return value

        for path in stage.rglob("*.json"):
            write_json(path, relocate(json.loads(path.read_text())))
        final = relocate(manifest)
        final["files"] = {
            str(path.relative_to(stage)): file_sha256(path)
            for path in stage.rglob("*")
            if path.is_file() and path != stage / "manifest.json"
        }
        write_json(stage / "manifest.json", final)
        return final
