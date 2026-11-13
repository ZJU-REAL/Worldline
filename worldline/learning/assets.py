"""Locate verified dataset media without depending on a trainer or simulator."""

from __future__ import annotations

import json
from pathlib import Path


def dataset_media_root(path: Path, explicit: Path | None = None) -> Path | None:
    if explicit is not None:
        return explicit.resolve(strict=True)
    path = path.resolve()
    candidates = (path.with_suffix(path.suffix + ".manifest.json"), path.parent / "split.manifest.json")
    for manifest in candidates:
        if manifest.is_file():
            data = json.loads(manifest.read_text())
            media = data.get("media") or {}
            if media.get("mode") == "reference" and media.get("root") is not None:
                return (manifest.parent / media["root"]).resolve(strict=True)
    for parent in path.parents:
        manifest = parent / "manifest.json"
        if not manifest.is_file():
            continue
        data = json.loads(manifest.read_text())
        if data.get("schema_version") == "worldline.distillation_bundle/0.1":
            return parent
    return None
