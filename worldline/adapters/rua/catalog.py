"""Read RUA handoff and EFE task trees into a small, portable task index."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from worldline.adapters.harbor import load_task
from worldline.run import write_json
from worldline.storage import file_sha256


def build_catalog(
    root: Path, *, mode: str | None = None, split: str | None = None, benchmark: str | None = None
) -> dict[str, Any]:
    root = root.resolve()
    task_root = root / "tasks" if (root / "tasks").is_dir() else root
    datasets = {}
    portable = root / "portable-catalog.json"
    if portable.is_file():
        payload = json.loads(portable.read_text(encoding="utf-8"))
        datasets = {row["id"]: row for row in payload.get("datasets", [])}
    entries, errors = [], []
    search_root = task_root
    if benchmark:
        matches = [path for path in task_root.iterdir() if path.is_dir() and path.name == benchmark]
        if not matches:
            return {
                "schema_version": "worldline.catalog/0.1",
                "domain": "rua",
                "root": str(root),
                "tasks": [],
                "count": 0,
                "errors": [],
            }
        search_root = matches[0]
    for manifest in sorted(search_root.rglob("task.toml")):
        task_dir = manifest.parent.resolve()
        if not task_dir.is_relative_to(root):
            continue
        try:
            task = load_task(task_dir)
            pool = task_dir.parent.name
            source = datasets.get(pool, {})
            eligible = source.get("eligible_tasks")
            if isinstance(eligible, list) and task_dir.name not in eligible:
                continue
            metadata = task.metadata
            task_mode = metadata.get("mode", source.get("mode", "unspecified"))
            task_split = metadata.get("split", source.get("split", "unspecified"))
            if mode and task_mode != mode or split and task_split != split or benchmark and pool != benchmark:
                continue
            # Whitelist metadata: local catalog contents, credentials and private
            # publication/request URLs never become public framework source.
            entries.append(
                {
                    "id": task.id,
                    "path": str(task_dir.relative_to(root)),
                    "backend": "harbor",
                    "domain": "rua",
                    "benchmark": pool,
                    "version": task.version,
                    "mode": task_mode,
                    "split": task_split,
                    "seed": task.seed,
                    "split_group": metadata.get("split_group", metadata.get("family", task.id)),
                    "resources": metadata.get("resources", {}),
                    "ready_for_collection": source.get("ready_for_collection"),
                    "training_sample_loader_verified": source.get("readiness", {}).get(
                        "training_sample_loader_verified"
                    ),
                }
            )
        except (OSError, ValueError, KeyError) as error:
            errors.append({"path": str(task_dir.relative_to(root)), "error": str(error)})
    return {
        "schema_version": "worldline.catalog/0.1",
        "domain": "rua",
        "root": str(root),
        "tasks": entries,
        "count": len(entries),
        "errors": errors,
    }


def write_catalog(root: Path, output: Path, **filters) -> dict[str, Any]:
    catalog = build_catalog(root, **filters)
    write_json(output, catalog)
    return catalog


def catalog_task_paths(catalog_path: Path) -> list[Path]:
    data = json.loads(catalog_path.read_text(encoding="utf-8"))
    if data.get("schema_version") != "worldline.catalog/0.1":
        raise ValueError("unsupported catalog schema")
    root = Path(data["root"]).resolve()
    paths = []
    for task in data["tasks"]:
        path = (root / task["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("catalog task path escapes its root")
        loaded = load_task(path)
        if loaded.id != task["id"] or loaded.version != task["version"]:
            raise ValueError(f"catalog identity changed: {task['id']}")
        paths.append(path)
    return paths


def catalog_task_metadata(catalog_path: Path) -> dict[Path, dict]:
    paths = catalog_task_paths(catalog_path)
    data = json.loads(catalog_path.read_text(encoding="utf-8"))
    revision = file_sha256(catalog_path)
    metadata = {}
    for path, entry in zip(paths, data["tasks"], strict=True):
        loaded = load_task(path)
        if entry.get("seed", loaded.seed) != loaded.seed:
            raise ValueError("catalog seed differs from the underlying task")
        split = entry.get("split", "unspecified")
        if loaded.split != "unspecified" and split != loaded.split:
            raise ValueError("catalog cannot override a task's declared split")
        metadata[path] = {
            "split": split,
            "split_group": entry.get("split_group", loaded.id),
            "catalog_sha256": revision,
        }
    return metadata
