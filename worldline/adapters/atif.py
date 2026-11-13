"""ATIF chat and media conversion, independent of Harbor's Python package."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Mapping
from worldline.media import MediaStore

IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}


def _content(
    value: Any, root: Path | None, trajectory_path: Path | None, media_store: MediaStore | None
) -> str | list[dict]:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise ValueError("ATIF content must be text or content parts")
    parts = []
    for part in value:
        if isinstance(part, str):
            parts.append({"type": "text", "text": part})
        elif isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
            parts.append({"type": "text", "text": part["text"]})
        elif isinstance(part, dict) and part.get("type") == "image":
            source = part.get("source") or {}
            mime, location = source.get("media_type"), source.get("path")
            if mime not in IMAGE_TYPES or not isinstance(location, str):
                raise ValueError("invalid ATIF image source")
            if root is None or trajectory_path is None:
                raise ValueError("ATIF media requires its trusted artifact root and trajectory path")
            if "://" in location or location.startswith("data:"):
                raise ValueError("ATIF media must be archived locally; remote media is not fetched during export")
            root = root.resolve()
            candidates = {(trajectory_path.parent / location).resolve(), (root / location).resolve()}
            if any(not path.is_relative_to(root) for path in candidates):
                raise ValueError("ATIF media path escapes trusted artifact root")
            existing = [path for path in candidates if path.is_file()]
            if len(existing) != 1:
                raise ValueError("missing or ambiguous ATIF image reference")
            path = existing[0]
            if path.stat().st_size > 32 * 1024 * 1024:
                raise ValueError("ATIF image exceeds 32 MiB per-image export limit")
            url = (
                media_store.import_file(path, mime, "data_url")
                if media_store
                else f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()
            )
            parts.append({"type": "image_url", "image_url": {"url": url}})
        else:
            # An unsupported modality must be visible as rejected data, never
            # silently flattened into a filename or an 'embedded' text token.
            raise ValueError("unsupported ATIF content part; image/text export only")
    return parts


def _tool_call(call: Mapping) -> dict:
    if "function" in call:
        return dict(call)
    return {
        "id": call["tool_call_id"],
        "type": "function",
        "function": {
            "name": call["function_name"],
            "arguments": json.dumps(call.get("arguments", {}), allow_nan=False),
        },
    }


def atif_messages(
    trajectory: Mapping[str, Any],
    *,
    artifact_root: Path | None = None,
    trajectory_path: Path | None = None,
    media_store: MediaStore | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Preserve actual text/image content and tool observations for training."""
    messages = []
    role_map = {"agent": "assistant", "user": "user", "system": "system", "environment": "tool"}
    for step in trajectory.get("steps") or []:
        if not isinstance(step, dict):
            raise ValueError("invalid ATIF step")
        if step.get("is_copied_context"):
            raise ValueError("copied ATIF context requires a trainer with explicit loss masking")
        role = role_map.get(str(step.get("source")), str(step.get("source") or "user"))
        message = {"role": role, "content": _content(step.get("message"), artifact_root, trajectory_path, media_store)}
        calls = step.get("tool_calls")
        if calls:
            message["tool_calls"] = [_tool_call(call) for call in calls]
        if message["content"] or calls:
            messages.append(message)
        observation = step.get("observation")
        if not isinstance(observation, dict):
            continue
        for result in observation.get("results") or []:
            if not isinstance(result, dict):
                raise ValueError("invalid ATIF observation result")
            tool = {
                "role": "tool",
                "content": _content(result.get("content"), artifact_root, trajectory_path, media_store),
            }
            if result.get("source_call_id"):
                tool["tool_call_id"] = result["source_call_id"]
            if tool["content"] or tool.get("tool_call_id"):
                messages.append(tool)
    return tuple(messages)
