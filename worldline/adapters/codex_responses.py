"""Stateless Responses framing around model decisions; Codex retains its agent loop.

This adapter never executes a tool or awards a reward. It translates only the
four public environment tools and explicitly records its prompting mode.
"""

import json
import time
import uuid

from worldline.agents.tool_agent import OpenAICompatibleAgent
from worldline.errors import AgentError, BudgetExceeded, InvalidAction
from worldline.harnesses.json_agent import parse_action


PUBLIC_TOOLS = {"observe", "act", "reset", "finish"}


def public_name(name):
    if not isinstance(name, str):
        return None
    if name in PUBLIC_TOOLS:
        return name
    for prefix in ("mcp__worldline__", "mcp_worldline_"):
        if name.startswith(prefix) and name[len(prefix) :] in PUBLIC_TOOLS:
            return name[len(prefix) :]
    return None


def content_for_chat(content):
    if isinstance(content, str) or content is None:
        return content
    if not isinstance(content, list):
        return json.dumps(content, ensure_ascii=False)
    result = []
    for part in content:
        kind = part.get("type")
        if kind in {"input_text", "output_text", "text"}:
            result.append({"type": "text", "text": part.get("text", "")})
        elif kind == "input_image":
            url = part.get("image_url")
            if not isinstance(url, str) or not url.startswith("data:image/"):
                raise InvalidAction("only inline public observation images are supported")
            result.append({"type": "image_url", "image_url": {"url": url}})
        else:
            raise InvalidAction(f"unsupported Responses content: {kind}")
    return result


def messages_for_chat(body):
    messages = []
    if body.get("instructions"):
        messages.append({"role": "system", "content": body["instructions"]})
    inputs = body.get("input", [])
    if isinstance(inputs, str):
        inputs = [{"role": "user", "content": inputs}]
    if not isinstance(inputs, list):
        raise InvalidAction("Responses input must be a list or string")
    for item in inputs:
        kind = item.get("type", "message")
        if kind == "message":
            role = item.get("role")
            if role not in {"user", "assistant", "system", "developer"}:
                raise InvalidAction("unsupported Responses message role")
            messages.append(
                {"role": "system" if role == "developer" else role, "content": content_for_chat(item.get("content"))}
            )
        elif kind == "function_call":
            name = item.get("name", "")
            if item.get("namespace"):
                name = item["namespace"] + "__" + name
            if not public_name(name):
                raise InvalidAction("Responses history contains a non-public tool")
            call = {
                "id": item["call_id"],
                "type": "function",
                "function": {"name": name, "arguments": item["arguments"]},
            }
            if messages and messages[-1]["role"] == "assistant":
                messages[-1].setdefault("tool_calls", []).append(call)
            else:
                messages.append({"role": "assistant", "content": None, "tool_calls": [call]})
        elif kind == "function_call_output":
            messages.append(
                {"role": "tool", "tool_call_id": item["call_id"], "content": content_for_chat(item.get("output"))}
            )
        elif kind != "reasoning":
            raise InvalidAction(f"unsupported Responses input item: {kind}")
    return messages


def public_tools(body):
    result = []
    for tool in body.get("tools", []):
        if tool.get("type") == "namespace" and tool.get("name") == "mcp__worldline":
            nested = [{**item, "name": tool["name"] + "__" + item["name"]} for item in tool.get("tools", [])]
            result.extend(public_tools({"tools": nested}))
            continue
        if tool.get("type") == "function" and public_name(tool.get("name", "")):
            result.append(
                {
                    "type": "function",
                    "function": {key: tool[key] for key in ("name", "description", "parameters") if key in tool},
                }
            )
    return result


def response_for(message, usage):
    output = []
    for call in message.get("tool_calls") or []:
        function = call["function"]
        if not public_name(function["name"]):
            raise AgentError("model requested a tool outside the public environment boundary")
        output.append(
            {
                "type": "function_call",
                "id": "fc_" + uuid.uuid4().hex,
                "call_id": call["id"],
                "name": function["name"],
                "arguments": function["arguments"],
                "status": "completed",
                **({"namespace": function["namespace"]} if function.get("namespace") else {}),
            }
        )
    if not output:
        text = message.get("content") or ""
        if not isinstance(text, str):
            raise AgentError("assistant final response must be text")
        output.append(
            {
                "type": "message",
                "id": "msg_" + uuid.uuid4().hex,
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        )
    return {
        "id": "resp_" + uuid.uuid4().hex,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "output": output,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", usage.get("input_tokens", 0)),
            "output_tokens": usage.get("completion_tokens", usage.get("output_tokens", 0)),
            "total_tokens": usage.get("total_tokens", 0),
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


def sse_events(response):
    sequence = 0

    def event(kind, **payload):
        nonlocal sequence
        value = {"type": kind, "sequence_number": sequence, **payload}
        sequence += 1
        return f"event: {kind}\ndata: {json.dumps(value, ensure_ascii=False)}\n\n".encode()

    pending = {**response, "status": "in_progress", "output": [], "usage": None}
    yield event("response.created", response=pending)
    yield event("response.in_progress", response=pending)
    for index, item in enumerate(response["output"]):
        start = {**item, "status": "in_progress"}
        if item["type"] == "function_call":
            start["arguments"] = ""
        else:
            start["content"] = []
        yield event("response.output_item.added", output_index=index, item=start)
        if item["type"] == "function_call":
            yield event(
                "response.function_call_arguments.delta",
                output_index=index,
                item_id=item["id"],
                delta=item["arguments"],
            )
            yield event(
                "response.function_call_arguments.done",
                output_index=index,
                item_id=item["id"],
                arguments=item["arguments"],
            )
        else:
            part = item["content"][0]
            yield event(
                "response.content_part.added",
                output_index=index,
                item_id=item["id"],
                content_index=0,
                part={**part, "text": ""},
            )
            yield event(
                "response.output_text.delta",
                output_index=index,
                item_id=item["id"],
                content_index=0,
                delta=part["text"],
            )
            yield event(
                "response.output_text.done", output_index=index, item_id=item["id"], content_index=0, text=part["text"]
            )
            yield event(
                "response.content_part.done", output_index=index, item_id=item["id"], content_index=0, part=part
            )
        yield event("response.output_item.done", output_index=index, item=item)
    yield event("response.completed", response=response)


class ChatResponsesAdapter(OpenAICompatibleAgent):
    def __init__(self, *, decision_mode="json", structured=False, **kwargs):
        super().__init__(**kwargs)
        self.decision_mode, self.structured = decision_mode, structured
        self.current_tools = []
        self.turns = 0

    def _tool_request(self):
        if self.decision_mode == "tools":
            return {"tools": self.current_tools, "tool_choice": "auto"} if self.current_tools else {}
        if self.structured:
            alternatives = []
            for tool in self.current_tools:
                function = tool["function"]
                parameters = function.get("parameters", {"type": "object"})
                if public_name(function["name"]) == "act":
                    parameters = {
                        "type": "object",
                        "properties": {"action": self.spec["action_schema"]},
                        "required": ["action"],
                        "additionalProperties": False,
                    }
                alternatives.append(
                    {
                        "type": "object",
                        "properties": {"tool": {"const": function["name"]}, "arguments": parameters},
                        "required": ["tool", "arguments"],
                        "additionalProperties": False,
                    }
                )
            alternatives.append(
                {
                    "type": "object",
                    "properties": {
                        "tool": {"const": "final"},
                        "arguments": {"type": "object", "properties": {}, "additionalProperties": False},
                        "text": {"type": "string"},
                    },
                    "required": ["tool", "arguments", "text"],
                    "additionalProperties": False,
                }
            )
            schema = {"oneOf": alternatives}
            return {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "worldline_decision", "schema": schema},
                }
            }
        return {}

    def decide(self, body):
        if self.turns >= self.max_turns:
            raise BudgetExceeded("Codex model turn budget exhausted")
        self.turns += 1
        self.current_tools = public_tools(body)
        messages = messages_for_chat(body)
        if self.decision_mode == "json":
            # Canonical records preserve the actual adapted prompt seen by the model.
            descriptions = [t["function"] for t in self.current_tools]
            instructions = []
            for message in messages:
                if message["role"] == "system":
                    content = message.get("content") or ""
                    instructions.append(
                        content
                        if isinstance(content, str)
                        else "\n".join(p["text"] for p in content if p.get("type") == "text")
                    )
            messages = [message for message in messages if message["role"] != "system"]
            instructions.append(
                'Return exactly one JSON object {"tool":"<tool name>","arguments":{...}}. To end use {"tool":"final","arguments":{},"text":"..."}. No prose outside JSON. Available tools: '
                + json.dumps(descriptions, ensure_ascii=False)
            )
            messages.insert(0, {"role": "system", "content": "\n\n".join(instructions)})
            for message in messages:
                if message.get("tool_calls"):
                    calls = message.pop("tool_calls")
                    decisions = [
                        {"tool": call["function"]["name"], "arguments": json.loads(call["function"]["arguments"])}
                        for call in calls
                    ]
                    message["content"] = json.dumps(decisions[0] if len(decisions) == 1 else decisions)
                if message["role"] == "tool":
                    message["role"] = "user"
                    message.pop("tool_call_id", None)
        self.messages[:] = messages
        before = dict(self.usage)
        message = self._completion()
        if message.get("role") != "assistant":
            raise AgentError("model response must have assistant role")
        self.messages.append(message)
        self.emit("agent.message", {"message": message})
        if self.decision_mode == "json":
            text = message.get("content")
            try:
                value = json.loads(text) if isinstance(text, str) else {}
                if value.get("tool") == "final" and isinstance(value.get("text", ""), str):
                    message = {"role": "assistant", "content": value.get("text", "")}
                else:
                    decision = {key: value[key] for key in ("tool", "arguments") if key in value}
                    if not set(value) <= {"tool", "arguments", "text"}:
                        raise InvalidAction("extra decision fields")
                    name, arguments = parse_action(json.dumps(decision))
                    if name not in {t["function"]["name"] for t in self.current_tools}:
                        raise InvalidAction("model selected an unavailable tool")
                    message = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_" + uuid.uuid4().hex,
                                "type": "function",
                                "function": {"name": name, "arguments": json.dumps(arguments)},
                            }
                        ],
                    }
            except (ValueError, AttributeError, InvalidAction):
                # Invalid model output is an agent decision, not an API fault.
                self.emit("adapter.invalid_decision", {})
                message = {"role": "assistant", "content": text if isinstance(text, str) else "Invalid model decision"}
        delta = {
            key: value - before.get(key, 0) for key, value in self.usage.items() if isinstance(value, (int, float))
        }
        namespace_mode = any(
            tool.get("type") == "namespace" and tool.get("name") == "mcp__worldline" for tool in body.get("tools", [])
        )
        if namespace_mode:
            for call in message.get("tool_calls") or []:
                function = call["function"]
                if function["name"].startswith("mcp__worldline__"):
                    function["namespace"] = "mcp__worldline"
                    function["name"] = function["name"].removeprefix("mcp__worldline__")
        return response_for(message, delta)


class ResponsesModelAdapter(ChatResponsesAdapter):
    def __init__(self, **kwargs):
        if kwargs.get("sampling_seed") is not None or kwargs.get("temperature", 0) != 0 or kwargs.get("top_p", 1) != 1:
            raise ValueError(
                "Responses transport uses provider-default sampling; use Chat transport for sampling overrides"
            )
        kwargs["decision_mode"] = "tools"
        super().__init__(**kwargs)
        self.url = self.base_url + "/responses"

    @property
    def worldline_config(self):
        return {
            **super().worldline_config,
            "temperature": None,
            "top_p": None,
            "sampling_seed": None,
            "sampling_policy": "provider_default",
        }

    def _request_body(self, context, tool_request, used):
        inputs = []
        for message in context:
            if message["role"] == "tool":
                inputs.append(
                    {"type": "function_call_output", "call_id": message["tool_call_id"], "output": message["content"]}
                )
                continue
            if message.get("content") is not None:
                content = message["content"]
                if isinstance(content, list):
                    content = [
                        {"type": "input_text", "text": p["text"]}
                        if p["type"] == "text"
                        else {"type": "input_image", "image_url": p["image_url"]["url"]}
                        for p in content
                    ]
                inputs.append({"role": message["role"], "content": content})
            for call in message.get("tool_calls") or []:
                inputs.append(
                    {
                        "type": "function_call",
                        "call_id": call["id"],
                        "name": call["function"]["name"],
                        "arguments": call["function"]["arguments"],
                    }
                )
        tools = [{"type": "function", **tool["function"]} for tool in self.current_tools]
        return {
            "model": self.model,
            "input": inputs,
            "tools": tools,
            "stream": False,
            "max_output_tokens": min(self.max_output_tokens, self.max_total_tokens - int(used))
            if self.max_total_tokens
            else self.max_output_tokens,
        }

    def _response_message(self, result):
        if result.get("status") in {"failed", "cancelled"}:
            raise ValueError("Responses generation failed")
        outputs = result["output"]
        text, calls = [], []
        for item in outputs:
            if item["type"] == "message":
                text.extend(p["text"] for p in item.get("content", []) if p.get("type") == "output_text")
            elif item["type"] == "function_call":
                calls.append(
                    {
                        "type": "function",
                        "id": item["call_id"],
                        "function": {"name": item["name"], "arguments": item["arguments"]},
                    }
                )
        if not text and not calls:
            raise ValueError("Responses returned no message or tool call")
        return {"role": "assistant", "content": "\n".join(text) or None, **({"tool_calls": calls} if calls else {})}
