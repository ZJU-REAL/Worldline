"""LLaMA-Factory ShareGPT supervision and role mapping."""

import json
from worldline.learning.formats.messages import normalized_messages


def llama_factory_record(row, media, *, tool_prose_policy="preserve", **options):
    if tool_prose_policy not in {"preserve", "reject", "omit"}:
        raise ValueError("tool_prose_policy must be preserve, reject or omit")
    messages, images = normalized_messages(row, media, **options)
    conversations, system, dropped_thoughts = [], [], 0
    for message in messages:
        role, content = message["role"], message["content"]
        if role == "system":
            if conversations:
                raise ValueError("mid-episode system messages require a custom trainer")
            system.append(content)
            continue
        if message.get("tool_calls"):
            if content and tool_prose_policy == "reject":
                raise ValueError(
                    "LLaMA-Factory function supervision cannot preserve simultaneous assistant prose; choose explicit omit policy or use canonical messages"
                )
            dropped_thoughts += int(bool(content) and tool_prose_policy == "omit")
            calls = [
                {"name": call["function"]["name"], "arguments": json.loads(call["function"]["arguments"])}
                for call in message["tool_calls"]
            ]
            if content and tool_prose_policy == "preserve":
                calls[0]["_worldline_assistant_prose"] = content
            content = json.dumps(calls, ensure_ascii=False)
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
