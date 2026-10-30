"""Build a reproducible training bundle from normalized evaluation runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from worldline.datasets import export_dataset, split_dataset, validate_datasets
from worldline.report import write_report
from worldline.run import write_json


def build_distillation_bundle(
    inputs: Sequence[Path],
    output_dir: Path,
    *,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: str = "worldline-v1",
    include_failures_in_sft: bool = False,
    max_pairs_per_task: int = 50,
) -> dict[str, Any]:
    """Export, split, and validate all supported training views."""

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = write_report(inputs, output_dir / "report.json")
    datasets: dict[str, Any] = {}

    for format_name in ("sft", "preference", "rl"):
        raw_path = output_dir / "raw" / f"{format_name}.jsonl"
        export = export_dataset(
            inputs,
            raw_path,
            format=format_name,  # type: ignore[arg-type]
            include_failures=include_failures_in_sft,
            max_pairs_per_task=max_pairs_per_task,
        )
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
