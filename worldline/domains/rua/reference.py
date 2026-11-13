"""Development policies for validating the MuJoCo adapter and data pipeline."""

from __future__ import annotations

import json
import math
import time

from worldline.agents.tool_agent import observation_content, prepare_messages
from worldline.contracts import Sandbox, Task
from worldline.errors import BudgetExceeded


class ReachReferenceAgent:
    @property
    def worldline_config(self):
        return {"reference_policy": "planar-inverse-kinematics", "noop": self.noop}

    def __init__(self, *, spec, initial_observation, emit, messages, usage, max_turns, deadline, noop=False):
        self.spec, self.initial = spec, initial_observation
        self.emit, self.messages = emit, messages
        self.max_turns, self.deadline, self.noop = max_turns, deadline, noop

    def run(self, task: Task, sandbox: Sandbox) -> None:
        if self.spec.get("robot") != "planar-two-link-arm":
            raise ValueError("reference policy supports only the MuJoCo reach fixture")
        prepare_messages(self.messages, task, self.spec, self.initial)
        if self.noop:
            self.messages.append({"role": "assistant", "content": "I end the attempt without acting."})
            return
        x, y = self.initial["target_position"]
        first, second = self.spec["link_lengths"]
        elbow = math.acos((x * x + y * y - first * first - second * second) / (2 * first * second))
        shoulder = math.atan2(y, x) - math.atan2(second * math.sin(elbow), first + second * math.cos(elbow))
        action = {"joint_targets": [shoulder, elbow]}
        for index in range(self.max_turns):
            if time.monotonic() >= self.deadline:
                raise BudgetExceeded("reference policy wall-clock budget exhausted")
            call_id = f"act-{index}"
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "act", "arguments": json.dumps({"action": action})},
                    }
                ],
            }
            self.messages.append(message)
            self.emit("agent.message", {"message": message})
            observation = sandbox.act(action)
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": observation_content(observation)})
            distance = math.hypot(*(a - b for a, b in zip(observation["tip_position"], [x, y])))
            if distance < 0.015 and math.hypot(*observation["joint_velocities"]) < 0.1:
                self.messages.append(
                    {"role": "assistant", "content": "The tip is at the target and motion has settled."}
                )
                return
        raise BudgetExceeded("reference policy turn budget exhausted")
