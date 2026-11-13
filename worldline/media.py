"""Content-addressed binary assets, shared by traces and training handoff."""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from typing import Any
from collections import OrderedDict
from worldline.storage import publish_bytes, publish_file, file_sha256

MARKER = "__worldline_media__"
EXTENSIONS = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}


class MediaStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.files_written = 0
        self.bytes_written = 0
        self.imported = OrderedDict()

    def import_file(self, path: Path, mime_type: str, encoding: str, *, sha256: str | None = None) -> dict:
        digest = sha256 or file_sha256(path)
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid media digest")
        if encoding not in {"data_url", "base64"}:
            raise ValueError("unsupported media encoding")
        relative = Path("media") / (digest + EXTENSIONS.get(mime_type, ".bin"))
        destination = self.root / relative
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, digest)
        if key not in self.imported:
            if destination.exists():
                if file_sha256(path) != digest or file_sha256(destination) != digest:
                    raise ValueError("media artifact hash mismatch")
            elif publish_file(path, destination, sha256=digest):
                self.files_written += 1
                self.bytes_written += stat.st_size
            self.imported[key] = True
            if len(self.imported) > 4096:
                self.imported.popitem(last=False)
        return {MARKER: {"path": str(relative), "sha256": digest, "mime_type": mime_type, "encoding": encoding}}

    def import_references(self, value: Any, origin: Path) -> Any:
        if isinstance(value, dict):
            if MARKER in value:
                description = value[MARKER]
                path = media_path(value, origin, verify=False)
                return self.import_file(
                    path, description["mime_type"], description["encoding"], sha256=description["sha256"]
                )
            return {key: self.import_references(nested, origin) for key, nested in value.items()}
        if isinstance(value, (tuple, list)):
            return [self.import_references(nested, origin) for nested in value]
        return self.externalize(value)

    def put(self, encoded: str, mime_type: str, encoding: str) -> dict:
        raw = base64.b64decode(encoded, validate=True)
        digest = hashlib.sha256(raw).hexdigest()
        relative = Path("media") / (digest + EXTENSIONS.get(mime_type, ".bin"))
        path = self.root / relative
        if not path.exists():
            if publish_bytes(path, raw, exclusive=True):
                self.files_written += 1
                self.bytes_written += len(raw)
        if file_sha256(path) != digest:
            raise ValueError("existing media artifact hash mismatch")
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

    def inventory(self) -> tuple[int, int]:
        count, size = 0, 0
        for path in (self.root / "media").glob("*"):
            if path.is_file() and not path.name.startswith("."):
                count += 1
                size += path.stat().st_size
        return count, size


def media_path(reference: dict, root: Path, *, verify: bool = True) -> Path:
    description = reference[MARKER]
    root = root.resolve()
    path = (root / description["path"]).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError("media path escapes trusted artifact root")
    if not path.is_file():
        raise ValueError("media reference is not a file")
    if verify and file_sha256(path) != description["sha256"]:
        raise ValueError("media artifact hash mismatch")
    return path


def media_bytes(reference: dict, root: Path) -> bytes:
    return media_path(reference, root).read_bytes()


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


def model_context(messages, *, root: Path | None = None, max_images: int | None = None, max_bytes=None, hydrate=True):
    """Hydrate only retained visual inputs; keep all text/tool message ordering."""
    if max_images is not None and (type(max_images) is not int or max_images < 0):
        raise ValueError("context image budget must be a nonnegative integer")
    remaining = max_images
    selected = []
    for message in reversed(messages):
        value = dict(message)
        if isinstance(value.get("content"), list):
            parts = []
            for part in reversed(value["content"]):
                if part.get("type") == "image_url" and remaining is not None:
                    if remaining == 0:
                        parts.append(
                            {"type": "text", "text": "Earlier image omitted by the configured context image budget."}
                        )
                        continue
                    remaining -= 1
                parts.append(part)
            value["content"] = list(reversed(parts))
        selected.append(value)
    result = list(reversed(selected))
    if max_bytes is not None:
        import json
        from worldline.agents.context import bounded_context

        def size(value):
            measured = len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode())
            if root:
                for message in value:
                    if isinstance(message.get("content"), list):
                        for part in message["content"]:
                            if part.get("type") == "image_url":
                                url = part["image_url"]["url"]
                                if isinstance(url, dict) and MARKER in url:
                                    measured += (
                                        4 * ((media_path(url, root, verify=False).stat().st_size + 2) // 3) + 100
                                    )
            return measured

        result, _ = bounded_context(result, max_bytes, measure=size)
    return hydrate_media(result, root) if root is not None and hydrate else result


class MessageHistory(list):
    """Append-compatible history that retains binary references between model calls."""

    def __init__(self, media: MediaStore):
        super().__init__()
        self.media = media

    def append(self, message):
        super().append(self.media.externalize(message))

    def extend(self, messages):
        super().extend(self.media.externalize(messages))

    def __setitem__(self, key, value):
        super().__setitem__(key, self.media.externalize(value))

    def model_context(self, max_images=None, max_bytes=None, hydrate=True):
        return model_context(self, root=self.media.root, max_images=max_images, max_bytes=max_bytes, hydrate=hydrate)
