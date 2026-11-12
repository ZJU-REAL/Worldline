"""Content-addressed binary assets, shared by traces and training handoff."""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from typing import Any

MARKER = "__worldline_media__"
EXTENSIONS = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}


class MediaStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.files_written = 0
        self.bytes_written = 0

    def put(self, encoded: str, mime_type: str, encoding: str) -> dict:
        raw = base64.b64decode(encoded, validate=True)
        digest = hashlib.sha256(raw).hexdigest()
        relative = Path("media") / (digest + EXTENSIONS.get(mime_type, ".bin"))
        path = self.root / relative
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation tolerates duplicate blobs across concurrent callers.
            try:
                with path.open("xb") as stream:
                    stream.write(raw)
                self.files_written += 1
                self.bytes_written += len(raw)
            except FileExistsError:
                pass
        return {MARKER: {"path": str(relative), "sha256": digest, "mime_type": mime_type, "encoding": encoding}}

    def externalize(self, value: Any) -> Any:
        if isinstance(value, str) and value.startswith("data:") and ";base64," in value:
            prefix, data = value.split(";base64,", 1)
            return self.put(data, prefix[5:], "data_url")
        if isinstance(value, dict):
            if MARKER in value:
                return value
            if isinstance(value.get("mime_type"), str) and isinstance(value.get("data"), str):
                return {**value, "data": self.put(value["data"], value["mime_type"], "base64")}
            return {key: self.externalize(nested) for key, nested in value.items()}
        if isinstance(value, (tuple, list)):
            return [self.externalize(nested) for nested in value]
        return value


def media_bytes(reference: dict, root: Path) -> bytes:
    description = reference[MARKER]
    root = root.resolve()
    path = (root / description["path"]).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError("media path escapes trusted artifact root")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != description["sha256"]:
        raise ValueError("media artifact hash mismatch")
    return raw


def hydrate_media(value: Any, root: Path) -> Any:
    if isinstance(value, dict):
        if MARKER in value:
            description = value[MARKER]
            encoded = base64.b64encode(media_bytes(value, root)).decode()
            return (
                f"data:{description['mime_type']};base64,{encoded}"
                if description["encoding"] == "data_url"
                else encoded
            )
        return {key: hydrate_media(nested, root) for key, nested in value.items()}
    if isinstance(value, list):
        return [hydrate_media(nested, root) for nested in value]
    return value
