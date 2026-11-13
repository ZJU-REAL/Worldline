"""LLaMA-Factory ShareGPT supervision and role mapping."""

import json
from worldline.learning.formats.messages import normalized_messages


def llama_factory_record(row, media):
    messages, images = normalized_messages(row, media)
    conversations, system, dropped_thoughts = [], [], 0
    for message in messages:
        role, content = message["role"], message["content"]
        if role == "system":
            if conversations:
                raise ValueError("mid-episode system messages require a custom trainer")
            system.append(content)
            continue
        if message.get("tool_calls"):
            dropped_thoughts += int(bool(content))
            content = json.dumps(
                [
                    {"name": call["function"]["name"], "arguments": json.loads(call["function"]["arguments"])}
                    for call in message["tool_calls"]
                ],
                ensure_ascii=False,
            )
            tag = "function_call"
        else:
            tag = {"user": "human", "assistant": "gpt", "tool": "observation"}[role]
        if conversations and tag in {"human", "observation"} and conversations[-1]["from"] in {"human", "observation"}:
            conversations[-1]["value"] += "\n" + content
        else:
            conversations.append({"from": tag, "value": content})
    trailing = 0
    while conversations and conversations[-1]["from"] in {"human", "observation"}:
        conversations.pop()
        trailing += 1
    images = images[: sum(message["value"].count("<image>") for message in conversations)]
    for index, message in enumerate(conversations):
        valid = {"human", "observation"} if index % 2 == 0 else {"gpt", "function_call"}
        if message["from"] not in valid:
            raise ValueError("LLaMA-Factory requires alternating input/output roles")
    if len(conversations) % 2:
        raise ValueError("incomplete LLaMA-Factory SFT conversation")
    return {
        "id": row["id"],
        "conversations": conversations,
        "system": "\n".join(system),
        "tools": json.dumps(row.get("tools") or [], ensure_ascii=False),
        "images": images,
        "worldline_metadata": {**row.get("metadata", {}), "trailing_inputs_omitted": trailing},
    }, dropped_thoughts
