"""Bind a resolved inference service to a native Harness without domain imports."""

from worldline.agents.tool_agent import OpenAICompatibleAgent
from worldline.harnesses.json_agent import JSONActionAgent


def create_agent(*, model, options, json_actions=False, **runtime):
    initial = runtime["initial_observation"]
    if isinstance(initial, dict) and initial.get("images") and not model.vision:
        raise ValueError("task exposes images; select a vision model and pass --vision")
    base = JSONActionAgent if json_actions else OpenAICompatibleAgent

    class BoundAgent(base):
        @property
        def worldline_config(self):
            return {**super().worldline_config, "base_url": model.identity_url, "model_source": model.settings}

    return BoundAgent(
        model=model.name,
        base_url=model.base_url,
        api_key_env=model.api_key_env,
        model_revision=model.revision,
        **options,
        **runtime,
    )
