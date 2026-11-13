"""Optional verl BaseTool adapter. Imported only inside a separate verl install."""

import asyncio
import base64
import io
import json
import uuid

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import ToolResponse
from worldline.learning.service import EpisodeClient


class WorldlineEnvironmentTool(BaseTool):
    """A per-call tool handle points to one trainer-created episode.

    Supply create_kwargs.episode_id through verl tools_kwargs on every turn.
    release drops only this handle. Final verifier reward is fetched separately
    by trusted trainer code, and is never returned in a model tool observation.
    """

    def __init__(self, config, tool_schema):
        super().__init__(config, tool_schema)
        self.client = EpisodeClient(config["endpoint"], config.get("api_key_env", "WORLDLINE_TRAINING_KEY"))
        self.instances = {}

    async def create(self, instance_id=None, **kwargs):
        episode = kwargs.get("create_kwargs", {}).get("episode_id")
        if not isinstance(episode, str) or not episode:
            raise ValueError("trainer must create one episode and supply create_kwargs.episode_id")
        instance_id = instance_id or uuid.uuid4().hex
        self.instances[instance_id] = episode
        return instance_id, ToolResponse()

    async def execute(self, instance_id, parameters, **kwargs):
        episode = self.instances[instance_id]
        operation = parameters.get("op")
        if operation not in {"observe", "act", "reset", "finish"} or set(parameters) != (
            {"op", "action"} if operation == "act" else {"op"}
        ):
            return ToolResponse(text="invalid environment operation"), 0.0, {}
        arguments = {"action": parameters["action"]} if operation == "act" else {}
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": instance_id,
                    "type": "function",
                    "function": {"name": operation, "arguments": json.dumps(arguments, allow_nan=False)},
                }
            ],
        }
        responses = await asyncio.to_thread(
            self.client.call, "advance", episode_id=episode, message=message, request_id=instance_id
        )
        text, images = [], []
        for response in responses:
            content = response["content"]
            if isinstance(content, str):
                text.append(content)
            else:
                from PIL import Image

                for part in content:
                    if part["type"] == "text":
                        text.append(part["text"])
                    elif part["type"] == "image_url":
                        raw = base64.b64decode(part["image_url"]["url"].split(",", 1)[1], validate=True)
                        with Image.open(io.BytesIO(raw)) as image:
                            images.append(image.copy())
        return ToolResponse(text="\n".join(text), image=images or None), 0.0, {"episode_id": episode}

    async def release(self, instance_id, **kwargs):
        self.instances.pop(instance_id, None)
