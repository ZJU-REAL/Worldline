"""Connect an already provisioned rua-robot/1.0 session to Worldline.

The owner provisions and pins the simulator separately. This adapter neither
starts retired RUA local runners nor imports their simulator/training code.
"""

from __future__ import annotations

import base64
import hashlib
import json
import socket
import uuid
from pathlib import Path
from typing import Any, Mapping

from worldline.contracts import RunResult, Task
from worldline.errors import EnvironmentError, InvalidAction


class RUASocketEnvironment:
    def __init__(self, task: Task, config: Mapping[str, Any]) -> None:
        self.socket_path = str(config["socket_path"])
        self.image_root = Path(config["image_root"]).resolve()
        self.initial = self._public(self._call("observe"))
        schema = self.initial["action_schema"]
        self._spec = {
            "protocol": "rua-robot/1.0",
            "source_action_schema": schema,
            "action_schema": {
                "type": "object",
                "properties": {
                    "actions": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 50,
                        "items": {
                            "type": "array",
                            "minItems": len(schema["low"]),
                            "maxItems": len(schema["low"]),
                            "items": {"type": "number"},
                        },
                    }
                },
                "required": ["actions"],
                "additionalProperties": False,
            },
            "capabilities": {"visual_observations": True},
            "source_revision": config["source_revision"],
        }

    def _call(self, method: str, params=None):
        request = {"id": uuid.uuid4().hex, "method": method, "params": params or {}}
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(120)
            connection.connect(self.socket_path)
            connection.sendall(json.dumps(request).encode() + b"\n")
            with connection.makefile("rb") as stream:
                line = stream.readline(8 * 1024 * 1024 + 1)
        if len(line) > 8 * 1024 * 1024:
            raise EnvironmentError("RUA response exceeds 8 MiB")
        response = json.loads(line)
        if "error" in response:
            error = response["error"]
            cls = InvalidAction if error.get("type") in {"ValueError", "KeyError"} else EnvironmentError
            raise cls(error.get("message", "RUA server error"))
        return response["result"]

    def _public(self, observation):
        frames = []
        for name, frame in observation.get("images", {}).items():
            path = Path(frame["path"])
            if not path.is_absolute():
                path = self.image_root / path
            path = path.resolve(strict=True)
            if not path.is_relative_to(self.image_root):
                raise EnvironmentError("camera path escapes trusted image root")
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != frame["sha256"]:
                raise EnvironmentError("camera artifact hash mismatch")
            frames.append({"name": name, "mime_type": "image/png", "data": base64.b64encode(raw).decode()})
        # Whitelist: no source paths, private evaluator values or state files.
        return {
            **{
                key: observation[key]
                for key in (
                    "instruction",
                    "state",
                    "simulation_steps",
                    "remaining_steps",
                    "action_schema",
                    "simulation_paused",
                    "episode_ended",
                )
                if key in observation
            },
            "images": frames,
        }

    def spec(self):
        return self._spec

    def reset(self, seed):
        raise InvalidAction("rua-robot/1.0 does not expose reset; provision a fresh session")

    def observe(self):
        return self._public(self._call("observe"))

    def step(self, action):
        if not isinstance(action, dict) or set(action) != {"actions"}:
            raise InvalidAction("expected {'actions': [action_vector, ...]}")
        return self._public(self._call("step", action))

    def verify(self):
        result = self._call("evaluate")
        if result.get("simulation_verified") is not True or type(result.get("success")) is not bool:
            raise EnvironmentError("RUA server returned no trusted native verdict")
        return RunResult(
            float(result["success"]),
            {"success": float(result["success"]), "sim_steps": float(result["simulation_steps"])},
        )

    def close(self):
        # The caller owns the external simulator; Worldline owns only its worker.
        pass
