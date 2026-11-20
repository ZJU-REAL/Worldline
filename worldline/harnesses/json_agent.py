"""Public-tool action loop for chat models without function-calling support."""

import json

from worldline.agents.tool_agent import OpenAICompatibleAgent, observation_content, prepare_messages
from worldline.errors import AgentError, BudgetExceeded, InvalidAction
from worldline.runtime.tools import execute_tool


def parse_action(content):
    if not isinstance(content, str):
        raise InvalidAction("return one JSON object with tool and arguments")
    text = content.strip()
    if text.startswith("```json\n") and text.endswith("\n```"):
        text = text[8:-4].strip()
    try:

        def invalid_constant(value):
            raise ValueError(f"nonfinite JSON constant: {value}")

        action = json.loads(text, parse_constant=invalid_constant)
    except ValueError as error:
        raise InvalidAction("return one JSON object with tool and arguments; no surrounding prose") from error
    if (
        not isinstance(action, dict)
        or set(action) != {"tool", "arguments"}
        or not isinstance(action["tool"], str)
        or not isinstance(action["arguments"], dict)
    ):
        raise InvalidAction('expected {"tool": "act", "arguments": {"action": ...}}')
    return action["tool"], action["arguments"]


class JSONActionAgent(OpenAICompatibleAgent):
    def _tool_request(self):
        return {}

    @property
    def worldline_config(self):
        return {**super().worldline_config, "action_protocol": "worldline.json-action/0.1"}

    def run(self, task, sandbox):
        prepare_messages(self.messages, task, self.spec, self.initial)
        system = dict(self.messages[0])
        system["content"] += (
            '\nRespond with exactly one JSON object: {"tool":"act","arguments":{"action":<action>}}. '
            'Other tools: {"tool":"observe","arguments":{}}, {"tool":"reset","arguments":{}}, '
            '{"tool":"finish","arguments":{}}. Use the action_schema above. No prose or tool_calls. '
            "You receive actual tool feedback in the next user message."
        )
        self.messages[0] = system
        for _ in range(self.max_turns):
            raw = self._completion()
            if raw.get("role") != "assistant":
                raise AgentError("model message must have assistant role")
            message = {"role": "assistant", "content": raw.get("content")}
            self.messages.append(message)
            self.emit("agent.message", {"message": message})
            try:
                if raw.get("tool_calls"):
                    raise InvalidAction("JSON Harness expects content, not function tool_calls")
                name, arguments = parse_action(message["content"])
                observation, finished = execute_tool(name, arguments, sandbox)
                content = observation_content(observation)
                self.emit("agent.json_action", {"tool": name, "arguments": arguments})
            except InvalidAction as error:
                content, finished = json.dumps({"error": str(error)}), False
            self.messages.append({"role": "user", "content": content})
            self.emit("agent.json_feedback", {"content": content})
            if finished:
                return
        raise BudgetExceeded("model turn budget exhausted")
