"""Shared chat/tool validation and verified multimodal conversion."""

import base64
import io
import json


def _content(content, media, images):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError("message content must be text or multimodal parts")
    parts = []
    for part in content:
        if part.get("type") == "text":
            parts.append(part["text"])
        elif part.get("type") == "image_url":
            url = part["image_url"]["url"]
            if not isinstance(url, str) or not url.startswith("data:image/") or ";base64," not in url:
                raise ValueError("image handoff requires embedded verified image bytes")
            prefix, encoded = url.split(";base64,", 1)
            from PIL import Image

            try:
                Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))).verify()
            except Exception as error:
                raise ValueError("invalid training image bytes") from error
            reference = media.put(encoded, prefix[5:], "data_url")["__worldline_media__"]
            images.append(reference["path"])
            parts.append("<image>")
        else:
            raise ValueError("unsupported multimodal content type")
    return "\n".join(parts)


def normalized_messages(row, media):
    images, messages = [], []
    pending_calls = set()
    seen_calls = set()
    for raw in row["messages"]:
        message = dict(raw)
        role = message.get("role")
        if pending_calls and role != "tool":
            raise ValueError("a new message appeared before pending tool responses")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError("unsupported message role")
        message["content"] = _content(message.get("content"), media, images)
        calls = []
        for call in message.get("tool_calls") or []:
            if "function" in call:
                call = dict(call)
            else:
                call = {
                    "id": call.get("tool_call_id"),
                    "type": "function",
                    "function": {"name": call["function_name"], "arguments": json.dumps(call.get("arguments", {}))},
                }
            if role != "assistant" or not isinstance(call.get("id"), str) or not call["id"] or call["id"] in seen_calls:
                raise ValueError("missing or duplicate tool call id")
            arguments = call["function"]["arguments"]
            if isinstance(arguments, str):
                json.loads(arguments)
            else:
                call["function"] = {**call["function"], "arguments": json.dumps(arguments)}
            pending_calls.add(call["id"])
            seen_calls.add(call["id"])
            calls.append(call)
        if calls:
            message["tool_calls"] = calls
        if role == "tool":
            ident = message.get("tool_call_id")
            if ident not in pending_calls:
                raise ValueError("tool response has no matching pending call")
            pending_calls.remove(ident)
        messages.append(message)
    if pending_calls:
        raise ValueError("trajectory ends with unanswered tool calls")
    if not messages or not any(message["role"] == "assistant" for message in messages):
        raise ValueError("SFT trajectory must contain assistant supervision")
    return messages, images
