"""Explicit trace quality filters; task success and split admission stay separate."""

from dataclasses import dataclass, asdict
import json


@dataclass(frozen=True)
class QualityPolicy:
    max_actions: int | None = None
    max_turns: int | None = None
    max_tool_error_fraction: float = 1.0
    max_samples_per_task: int | None = None

    def __post_init__(self):
        for value in (self.max_actions, self.max_turns, self.max_samples_per_task):
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError("quality limits must be positive integers")
        if not 0 <= self.max_tool_error_fraction <= 1:
            raise ValueError("tool error fraction must be in [0, 1]")

    def accepts(self, episode):
        for key, maximum in (("n_actions", self.max_actions), ("n_turns", self.max_turns)):
            if maximum is not None and (key not in episode.metrics or episode.metrics[key] > maximum):
                return False
        tools = [message for message in episode.messages if message.get("role") == "tool"]
        errors = 0
        for message in tools:
            content = message.get("content")
            if isinstance(content, str):
                try:
                    value = json.loads(content)
                    errors += isinstance(value, dict) and "error" in value
                except ValueError:
                    pass
        return not tools or errors / len(tools) <= self.max_tool_error_fraction

    def configuration(self):
        return asdict(self)
