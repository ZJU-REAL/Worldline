"""OpenAI-compatible chat/tool agent for native environments.

This client is for Worldline-native experiments. Official EFE runs continue to
use Harbor's native agents and usage/egress enforcement.
"""

from __future__ import annotations

import json
import os
import time
import random
import math
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlparse

from worldline.contracts import Sandbox, Task
from worldline.runtime.tools import execute_tool
from worldline.media import model_context, MessageHistory
from worldline.errors import AgentError, BudgetExceeded, InvalidAction, ModelAPIError


def tools_for(spec: dict) -> list[dict]:
    schemas = {
        "observe": {"type": "object", "properties": {}, "additionalProperties": False},
        "act": {
            "type": "object",
            "properties": {"action": spec["action_schema"]},
            "required": ["action"],
            "additionalProperties": False,
        },
        "reset": {"type": "object", "properties": {}, "additionalProperties": False},
        "finish": {"type": "object", "properties": {}, "additionalProperties": False},
    }
    descriptions = {
        "observe": "Read the current public environment observation.",
        "act": "Execute an action and read its actual outcome.",
        "reset": "Reset to the declared initial state, within the task's reset budget.",
        "finish": "End the attempt. The independent verifier judges the final state.",
    }
    return [
        {"type": "function", "function": {"name": name, "description": descriptions[name], "parameters": schema}}
        for name, schema in schemas.items()
    ]


def observation_content(observation: Any) -> str | list[dict]:
    if not isinstance(observation, dict) or not observation.get("images"):
        return json.dumps(observation, ensure_ascii=False)
    state = {key: value for key, value in observation.items() if key != "images"}
    content = [{"type": "text", "text": json.dumps(state, ensure_ascii=False)}]
    for frame in observation["images"]:
        if frame.get("name"):
            content.append({"type": "text", "text": f"Image: {frame['name']}"})
        content.append({"type": "image_url", "image_url": {"url": f"data:{frame['mime_type']};base64,{frame['data']}"}})
    return content


def prepare_messages(messages: list, task: Task, spec: dict, observation: Any) -> None:
    messages[:] = [
        {
            "role": "system",
            "content": (
                "Operate the environment using its public tools. Check feedback before claiming completion. "
                "A tool returning normally does not imply the task succeeded. Environment specification: "
                + json.dumps(spec, ensure_ascii=False)
            ),
        },
        {"role": "user", "content": task.instruction},
        {"role": "user", "content": observation_content(observation)},
    ]


class OpenAICompatibleAgent:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key_env: str = "WORLDLINE_API_KEY",
        spec: dict,
        initial_observation: Any,
        emit,
        messages: list,
        usage: dict,
        max_turns: int,
        deadline: float,
        max_retries: int = 2,
        retry_base_sec: float = 0.5,
        max_context_images: int | None = None,
    ) -> None:
        parsed = urlparse(base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("base_url must be an HTTP(S) URL without credentials, query or fragment")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.base_url = base_url.rstrip("/")
        self.model, self.api_key_env = model, api_key_env
        self.spec, self.initial = spec, initial_observation
        self.emit, self.messages, self.usage = emit, messages, usage
        self.max_turns, self.deadline = max_turns, deadline
        if type(max_retries) is not int or not 0 <= max_retries <= 5 or retry_base_sec < 0:
            raise ValueError("model retries must be in [0, 5] and retry delay nonnegative")
        self.max_retries, self.retry_base_sec = max_retries, retry_base_sec
        if max_context_images is not None and (type(max_context_images) is not int or max_context_images < 0):
            raise ValueError("max_context_images must be a nonnegative integer")
        self.max_context_images = max_context_images

    @property
    def worldline_config(self):
        return {
            "model": self.model,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "max_context_images": self.max_context_images,
            "max_retries": self.max_retries,
            "retry_base_sec": self.retry_base_sec,
        }

    def _completion(self) -> dict:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise BudgetExceeded("model wall-clock budget exhausted")
        body = {
            "model": self.model,
            "messages": self.messages.model_context(self.max_context_images)
            if isinstance(self.messages, MessageHistory)
            else model_context(self.messages, max_images=self.max_context_images),
            "tools": tools_for(self.spec),
            "tool_choice": "auto",
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self.api_key_env)
        if key:
            headers["Authorization"] = f"Bearer {key}"
        request = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers=headers)
        for attempt in range(self.max_retries + 1):
            self.usage["model_requests"] = self.usage.get("model_requests", 0) + 1
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise BudgetExceeded("model wall-clock budget exhausted")
            try:
                with urllib.request.urlopen(request, timeout=min(remaining, 120)) as response:
                    result = json.load(response)
                break
            except urllib.error.HTTPError as error:
                try:
                    retry_after = float(error.headers.get("Retry-After", ""))
                except (ValueError, TypeError):
                    retry_after = None
                failure = ModelAPIError(
                    f"model API returned HTTP {error.code}", status=error.code, retry_after=retry_after
                )
            except OSError as error:
                failure = ModelAPIError(f"model API transport failed: {type(error).__name__}")
            except ValueError as error:
                raise ModelAPIError("model API returned malformed JSON", status=422) from error
            if not failure.retryable or attempt == self.max_retries:
                raise failure
            delay = min(
                30,
                max(
                    0,
                    failure.retry_after
                    if failure.retry_after is not None
                    else self.retry_base_sec * 2**attempt * random.uniform(0.8, 1.2),
                ),
            )
            if delay >= self.deadline - time.monotonic():
                raise failure
            self.emit("model.retry", {"attempt": attempt + 1, "error_code": failure.code, "delay_sec": delay})
            time.sleep(delay)
        if not isinstance(result, dict):
            raise ModelAPIError("model API returned a non-object response", status=422)
        raw_usage = result.get("usage") or {}
        if not isinstance(raw_usage, dict):
            raise ModelAPIError("model API returned invalid usage metadata", status=422)
        safe_usage = {}
        for key, value in raw_usage.items():
            if type(value) in {int, float} and math.isfinite(value) and value >= 0:
                safe_usage[key] = value
                self.usage[key] = self.usage.get(key, 0) + value
            elif type(value) in {int, float, bool}:
                self.usage["invalid_usage_fields"] = self.usage.get("invalid_usage_fields", 0) + 1
        self.emit("model.response", {"model": self.model, "usage": safe_usage})
        try:
            message = result["choices"][0]["message"]
            if not isinstance(message, dict) or not isinstance(message.get("content"), (str, list, type(None))):
                raise ValueError("invalid chat message")
            json.dumps(message, allow_nan=False)
            return message
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise ModelAPIError("model API returned no valid chat message", status=422) from error

    def run(self, task: Task, sandbox: Sandbox) -> None:
        prepare_messages(self.messages, task, self.spec, self.initial)
        for _ in range(self.max_turns):
            message = self._completion()
            if message.get("role") != "assistant":
                raise AgentError("model message must have assistant role")
            message = {key: message[key] for key in ("role", "content", "tool_calls") if key in message}
            self.messages.append(message)
            self.emit("agent.message", {"message": message})
            calls = message.get("tool_calls") or []
            if not isinstance(calls, list) or len(calls) > 32:
                raise AgentError("model returned invalid or excessive tool calls")
            if not calls:
                return
            finished = False
            budget_error = None
            for call in calls:
                try:
                    call_id, function = call["id"], call["function"]
                    name = function["name"]
                    arguments = json.loads(function["arguments"])
                    if not isinstance(call_id, str) or not isinstance(arguments, dict):
                        raise ValueError("invalid tool call id or arguments")
                except (KeyError, TypeError, ValueError) as error:
                    raise AgentError(f"malformed tool call: {error}") from error
                try:
                    if finished:
                        raise InvalidAction("attempt already finished")
                    if budget_error is not None:
                        raise InvalidAction("attempt budget already exhausted")
                    result, finished = execute_tool(name, arguments, sandbox)
                    content = observation_content(result)
                except InvalidAction as error:
                    content = json.dumps({"error": str(error)})
                except BudgetExceeded as error:
                    budget_error = error
                    content = json.dumps({"error": str(error)})
                self.messages.append({"role": "tool", "tool_call_id": call_id, "content": content})
                self.emit("agent.tool_result", {"tool": name, "tool_call_id": call_id, "content": content})
            if budget_error is not None:
                raise budget_error
            if finished:
                return
        raise BudgetExceeded("model turn budget exhausted")
