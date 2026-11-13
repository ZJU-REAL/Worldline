"""Fixture-only CUA policy for checking real BrowserGym execution."""

import json
from worldline.agents.tool_agent import observation_content, prepare_messages


class FormReferenceAgent:
    @property
    def worldline_config(self):
        return {"reference_policy": "project-form-fixture", "noop": self.noop}

    def __init__(self, *, spec, initial_observation, emit, messages, usage, max_turns, deadline, noop=False):
        self.spec, self.initial, self.messages, self.emit, self.noop = spec, initial_observation, messages, emit, noop

    def run(self, task, sandbox):
        prepare_messages(self.messages, task, self.spec, self.initial)
        if self.noop:
            self.messages.append({"role": "assistant", "content": "I end this attempt without interacting."})
            return
        fields = json.loads(task.instruction.split("Fields: ", 1)[1].strip())
        actions = [
            {
                "op": "fill",
                "bid": next(
                    element["bid"]
                    for element in self.initial["elements"]
                    if element["role"] == "textbox" and element["name"] == name
                ),
                "text": value,
            }
            for name, value in fields.items()
        ]
        actions.append(
            {
                "op": "click",
                "bid": next(
                    element["bid"]
                    for element in self.initial["elements"]
                    if element["role"] == "button" and element["name"] == "Submit project"
                ),
            }
        )
        for index, action in enumerate(actions):
            call = {
                "id": f"cua-{index}",
                "type": "function",
                "function": {"name": "act", "arguments": json.dumps({"action": action})},
            }
            self.messages.append({"role": "assistant", "content": None, "tool_calls": [call]})
            observation = sandbox.act(action)
            self.messages.append(
                {"role": "tool", "tool_call_id": call["id"], "content": observation_content(observation)}
            )
        self.messages.append({"role": "assistant", "content": "The project form has been submitted."})
