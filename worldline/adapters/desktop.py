"""Desktop CUA through an independently installed OSWorld-compatible controller.

The controller owns VM/snapshot specifics. Worldline's public actions contain
structured mouse/keyboard/navigation data, never arbitrary shell or Python.
"""

from __future__ import annotations

import base64
import json
import urllib.request
import struct
import time
from urllib.parse import urlparse

from worldline.contracts import RunResult
from worldline.errors import EnvironmentError, InvalidAction
from worldline.runtime.provisioning import ExternalSession


class DesktopEnvironment:
    def __init__(self, task, config):
        self.session = ExternalSession(task, config)
        try:
            self.endpoint = self.session.descriptor["endpoint"].rstrip("/")
            parsed = urlparse(self.endpoint)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.netloc
                or parsed.username
                or parsed.password
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError("desktop endpoint must be a plain HTTP(S) URL")
            self.timeout = config.get("request_timeout_sec", 10)
            self.max_bytes = config.get("max_response_bytes", 16 * 1024 * 1024)
            if (
                type(self.max_bytes) is not int
                or self.max_bytes < 1
                or type(self.timeout) not in {int, float}
                or not 0 < self.timeout <= 120
            ):
                raise ValueError("invalid desktop byte budget or timeout")
            self.initial = self.observe()
        except BaseException:
            self.close()
            raise

    def _call(self, method, params=None):
        request = urllib.request.Request(
            self.endpoint + "/worldline/" + method,
            data=json.dumps(
                {"session_id": self.session.descriptor["session_id"], "params": params or {}}, allow_nan=False
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        from worldline.runtime.http import http_json

        result = http_json(request, time.monotonic() + self.timeout, self.max_bytes)
        if not isinstance(result, dict) or result.get("session_id") != self.session.descriptor["session_id"]:
            raise EnvironmentError("desktop response session identity mismatch")
        if result.get("error"):
            raise EnvironmentError("desktop controller failed")
        return result["result"]

    def spec(self):
        return {
            "domain": "cua",
            "protocol": "worldline.desktop/0.1",
            "capabilities": {"visual_observations": True},
            "action_schema": {
                "type": "object",
                "required": ["op"],
                "properties": {
                    "op": {"enum": ["click", "move", "drag", "scroll", "type", "key", "wait"]},
                    "x": {"type": "integer", "minimum": 0},
                    "y": {"type": "integer", "minimum": 0},
                    "to_x": {"type": "integer", "minimum": 0},
                    "to_y": {"type": "integer", "minimum": 0},
                    "text": {"type": "string"},
                    "key": {"type": "string"},
                    "amount": {"type": "integer"},
                    "seconds": {"type": "number", "minimum": 0, "maximum": 5},
                },
                "additionalProperties": False,
            },
            "source_revision": self.session.descriptor["source_revision"],
        }

    def observe(self):
        observation = self._call("observe")
        image = observation["screenshot_png_base64"]
        raw = base64.b64decode(image, validate=True)
        if len(raw) < 24 or raw[:8] != b"\x89PNG\r\n\x1a\n" or raw[12:16] != b"IHDR":
            raise EnvironmentError("desktop screenshot is not PNG")
        self.width, self.height = struct.unpack(">II", raw[16:24])
        if (
            self.width != observation["width"]
            or self.height != observation["height"]
            or not 0 < self.width * self.height <= 16_777_216
        ):
            raise EnvironmentError("desktop screenshot dimensions mismatch or exceed pixel budget")
        return {
            "images": [{"name": "desktop", "mime_type": "image/png", "data": image}],
            "width": observation["width"],
            "height": observation["height"],
            "episode_ended": observation.get("episode_ended", False),
        }

    def step(self, action):
        fields = {
            "click": {"x", "y"},
            "move": {"x", "y"},
            "drag": {"x", "y", "to_x", "to_y"},
            "scroll": {"amount"},
            "type": {"text"},
            "key": {"key"},
            "wait": {"seconds"},
        }
        if (
            not isinstance(action, dict)
            or action.get("op") not in fields
            or set(action) != {"op", *fields[action["op"]]}
        ):
            raise InvalidAction("invalid desktop operation or fields")
        for name in fields[action["op"]]:
            value = action[name]
            if name in {"text", "key"}:
                if not isinstance(value, str) or len(value) > 16384:
                    raise InvalidAction("invalid desktop text/key")
            elif name == "seconds":
                if type(value) not in {int, float} or not 0 <= value <= 5:
                    raise InvalidAction("desktop wait must be in [0, 5]")
            elif type(value) is not int or name != "amount" and value < 0:
                raise InvalidAction("desktop coordinates must be nonnegative integers")
            if name in {"x", "to_x"} and value >= self.width or name in {"y", "to_y"} and value >= self.height:
                raise InvalidAction("desktop coordinate is outside the viewport")
        self._call("act", action)
        return self.observe()

    def reset(self, seed):
        raise InvalidAction("desktop reset requires a newly provisioned session")

    def verify(self):
        result = self._call("verify")
        if result.get("verified") is not True or type(result.get("success")) is not bool:
            raise EnvironmentError("desktop controller returned no trusted verdict")
        return RunResult(
            float(result["success"]),
            {"success": float(result["success"]), "native_score": float(result.get("native_score", result["success"]))},
        )

    def execution_info(self):
        return {
            "session_id": self.session.descriptor["session_id"],
            "owned": self.session.owned,
            "identity_verified": self.session.verified,
        }

    def close(self):
        self.session.close()
