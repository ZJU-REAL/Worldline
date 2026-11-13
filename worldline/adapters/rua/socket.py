"""Connect an already provisioned rua-robot/1.0 session to Worldline.

The owner provisions and pins the simulator separately. This adapter neither
starts retired RUA local runners nor imports their simulator/training code.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import socket
import uuid
from pathlib import Path
from typing import Any, Mapping

from worldline.contracts import RunResult, Task
from worldline.errors import EnvironmentError, InvalidAction
from worldline.runtime.provisioning import ExternalSession


def managed_session(task, config):
    if not config.get("provider"):
        raise ValueError("rua.managed requires an owned provider for safe fresh-session retry")
    return RUASocketEnvironment(task, config)


class RUASocketEnvironment:
    def __init__(self, task: Task, config: Mapping[str, Any]) -> None:
        self.session = ExternalSession(task, config)
        try:
            self._initialize(config)
        except BaseException:
            self.session.close()
            raise

    def _initialize(self, config):
        self.include_state = config.get("include_state", True)
        if type(self.include_state) is not bool:
            raise ValueError("include_state must be boolean")
        self.socket_path = str(self.session.descriptor["socket_path"])
        self.image_root = Path(self.session.descriptor["image_root"]).resolve()
        self.max_image_bytes = config.get("max_image_bytes", 64 * 1024 * 1024)
        self.timeout_sec = config.get("request_timeout_sec", 10)
        if type(self.max_image_bytes) is not int or self.max_image_bytes < 1 or not 0 < self.timeout_sec <= 120:
            self.session.close()
            raise ValueError("invalid RUA image budget or request timeout")
        try:
            self.initial = self._public(self._call("observe"))
        except BaseException:
            self.session.close()
            raise
        schema = self.initial["action_schema"]
        if (
            not schema.get("low")
            or len(schema["low"]) != len(schema.get("high", []))
            or any(
                type(low) not in {int, float}
                or type(high) not in {int, float}
                or not math.isfinite(low)
                or not math.isfinite(high)
                or low > high
                for low, high in zip(schema["low"], schema["high"])
            )
        ):
            raise EnvironmentError("RUA service returned invalid action bounds")
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
            "capabilities": {"visual_observations": True, "privileged_state": bool(self.initial.get("state"))},
            "source_revision": config["source_revision"],
            "session_identity_verified": self.session.verified,
        }

    def _call(self, method: str, params=None):
        try:
            return self._rpc(method, params)
        except OSError as error:
            raise EnvironmentError(
                f"RUA transport failed: {type(error).__name__}", code="rua.transport", retryable=True
            ) from error

    def _rpc(self, method, params):
        request = {"id": uuid.uuid4().hex, "method": method, "params": params or {}}
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(self.timeout_sec)
            connection.connect(self.socket_path)
            connection.sendall(json.dumps(request).encode() + b"\n")
            with connection.makefile("rb") as stream:
                line = stream.readline(8 * 1024 * 1024 + 1)
        if len(line) > 8 * 1024 * 1024:
            raise EnvironmentError("RUA response exceeds 8 MiB")
        response = json.loads(line)
        if not isinstance(response, dict) or response.get("id") != request["id"]:
            raise EnvironmentError("RUA response request identity mismatch")
        if "error" in response:
            error = response["error"]
            cls = InvalidAction if error.get("type") in {"ValueError", "KeyError"} else EnvironmentError
            raise cls(error.get("message", "RUA server error"))
        return response["result"]

    def _public(self, observation):
        frames = []
        total_bytes = 0
        for name, frame in observation.get("images", {}).items():
            path = Path(frame["path"])
            if not path.is_absolute():
                path = self.image_root / path
            path = path.resolve(strict=True)
            if not path.is_relative_to(self.image_root):
                raise EnvironmentError("camera path escapes trusted image root")
            with path.open("rb") as stream:
                size = os.fstat(stream.fileno()).st_size
                total_bytes += size
                if total_bytes > self.max_image_bytes:
                    raise EnvironmentError("camera observation exceeds byte budget")
                raw = stream.read(size + 1)
                if len(raw) > size:
                    raise EnvironmentError("camera artifact changed during read")
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
                if key in observation and (key != "state" or self.include_state)
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
        actions = action["actions"]
        bounds = self._spec["source_action_schema"]
        if not isinstance(actions, list) or not 1 <= len(actions) <= 50:
            raise InvalidAction("RUA action batch must contain 1 to 50 vectors")
        for vector in actions:
            if (
                not isinstance(vector, list)
                or len(vector) != len(bounds["low"])
                or any(
                    type(value) not in {int, float} or not math.isfinite(value) or not low <= value <= high
                    for value, low, high in zip(vector, bounds["low"], bounds["high"])
                )
            ):
                raise InvalidAction("RUA action vector violates source bounds")
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
        self.session.close()

    def execution_info(self):
        return {
            "session_id": self.session.descriptor["session_id"],
            "owned": self.session.owned,
            "identity_verified": self.session.verified,
        }
