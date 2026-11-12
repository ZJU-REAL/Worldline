"""BrowserGym high-level/coordinate actions, observations and native validation."""

from __future__ import annotations

import base64
import html
import importlib
import importlib.util
import io
import json
import math
import os
import threading
import time
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from unittest.mock import patch

from worldline.contracts import RunResult
from worldline.errors import InvalidAction


class MemoryChat:
    """Noninteractive BrowserGym chat without a second Chromium UI."""

    def __init__(self, **kwargs):
        if kwargs.get("record_video_dir"):
            raise ValueError("in-memory chat does not record chat UI video")
        self.messages = []

    def add_message(self, role, msg):
        if role in {"user", "user_image", "assistant", "infeasible"}:
            self.messages.append({"role": role, "message": msg, "timestamp": time.time()})

    def close(self):
        pass


class FormServer:
    """A local CUA fixture with server-side submission verification."""

    def __init__(self, fields):
        self.fields, self.receipt = dict(fields), None
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                controls = "".join(
                    f'<label>{html.escape(name)}<input name="{html.escape(name, quote=True)}"></label><br>'
                    for name in owner.fields
                )
                page = f'<html><head><title>Worldline project form</title></head><body><h1>Create project</h1><form method="POST">{controls}<button>Submit project</button></form></body></html>'
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(page.encode())

            def do_POST(self):
                body = self.rfile.read(min(int(self.headers.get("Content-Length", 0)), 65536)).decode()
                submitted = parse_qs(body, keep_blank_values=True)
                owner.receipt = {name: submitted.get(name, [""])[0] for name in owner.fields}
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"<html><body><h1>Project submitted</h1></body></html>")

            def log_message(self, *_):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_port}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


class BrowserGymEnvironment:
    @staticmethod
    def preflight():
        spec = importlib.util.find_spec("playwright")
        if spec is None:
            raise ValueError("BrowserGym requires the cua extra and Playwright Chromium")
        if os.environ.get("WORLDLINE_CHROMIUM_PATH"):
            executable = Path(os.environ["WORLDLINE_CHROMIUM_PATH"])
        else:
            package = Path(next(iter(spec.submodule_search_locations)))
            browsers = json.loads((package / "driver/package/browsers.json").read_text())["browsers"]
            revision = next(browser["revision"] for browser in browsers if browser["name"] == "chromium")
            configured = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
            if configured == "0":
                cache = package / "driver/package/.local-browsers"
            elif configured:
                cache = Path(configured)
            else:
                cache = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "ms-playwright"
            executable = cache / f"chromium-{revision}" / "chrome-linux/chrome"
        if not executable.is_file():
            raise ValueError(
                "matching Chromium is not installed; run python -m playwright install chromium or set WORLDLINE_CHROMIUM_PATH"
            )

    @staticmethod
    def source_path():
        return __file__

    def __init__(self, task, config):
        import gymnasium as gym
        from browsergym.core.env import BrowserEnv
        from browsergym.core.task import AbstractBrowserTask
        from browsergym.core.action.highlevel import HighLevelActionSet

        self.fixture = None
        self.env = None
        self.render = config.get("render", True)
        self.navigation = config.get("allow_navigation", False)
        self.memory_chat = config.get("headless_chat", not bool(config.get("gym_id")))
        self.cpu_headless = config.get("cpu_headless", not bool(config.get("gym_id")))
        self.action_set = HighLevelActionSet(
            subsets=["bid", "coord", "chat"] + (["nav"] if self.navigation else []), multiaction=False, strict=True
        )
        browser_options = (
            {"executable_path": os.environ["WORLDLINE_CHROMIUM_PATH"]}
            if os.environ.get("WORLDLINE_CHROMIUM_PATH")
            else {}
        )
        try:
            if config.get("gym_id"):
                for module in config.get("benchmark_modules", []):
                    if not module.startswith("browsergym."):
                        raise ValueError("benchmark modules must belong to browsergym")
                    importlib.import_module(module)
                self.env = gym.make(
                    config["gym_id"],
                    headless=True,
                    action_mapping=self.action_set.to_python_code,
                    pw_chromium_kwargs=browser_options,
                    **config.get("task_kwargs", {}),
                )
            else:
                self.fixture = FormServer(config.get("fields", {"Project": "Orion", "Owner": "Chen"}))
                fixture, instruction = self.fixture, task.instruction

                class LocalFormTask(AbstractBrowserTask):
                    def setup(self, page):
                        fixture.receipt = None
                        page.goto(fixture.url)
                        return instruction, {}

                    def validate(self, page, chat_messages):
                        submitted = fixture.receipt is not None
                        return float(submitted and fixture.receipt == fixture.fields), submitted, "", {}

                self.env = BrowserEnv(
                    task_entrypoint=LocalFormTask,
                    headless=True,
                    slow_mo=0,
                    pre_observation_delay=0,
                    action_mapping=self.action_set.to_python_code,
                    pw_chromium_kwargs=browser_options,
                )
            self.reset(task.seed)
        except BaseException:
            self.close()
            raise

    def spec(self):
        return {
            "protocol": "worldline.browsergym/0.1",
            "domain": "cua",
            "browser": "Chromium",
            "browser_version": self.env.unwrapped.browser.version,
            "chat_transport": "in-memory" if self.memory_chat else "browser-ui",
            "chromium_mode": "headless-new-cpu" if self.cpu_headless else "browsergym-default",
            "action_schema": {
                "type": "object",
                "properties": {
                    "op": {
                        "enum": ["click", "fill", "press", "scroll", "mouse_click", "keyboard_type", "send_msg_to_user"]
                        + (["goto", "go_back"] if self.navigation else [])
                    },
                    "bid": {"type": "string"},
                    "text": {"type": "string"},
                    "key": {"type": "string"},
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "url": {"type": "string"},
                },
                "required": ["op"],
                "additionalProperties": False,
            },
            "action_semantics": "click/fill/press use BrowserGym bid; scroll uses x/y deltas; mouse_click uses viewport pixels",
            "capabilities": {"visual_observations": self.render},
            "fixture": "project-form" if self.fixture else None,
        }

    def _public(self):
        from browsergym.utils.obs import flatten_axtree_to_str

        raw = self.raw
        elements = [
            {
                "bid": str(node["browsergym_id"]),
                "role": node.get("role", {}).get("value"),
                "name": node.get("name", {}).get("value", ""),
            }
            for node in raw["axtree_object"]["nodes"]
            if node.get("browsergym_id")
        ]
        url = raw.get("url", "")
        if self.fixture:
            url = url.replace(self.fixture.url, "worldline://project-form")
        observation = {
            "url": url,
            "axtree": flatten_axtree_to_str(raw["axtree_object"]),
            "elements": elements,
            "last_action_error": raw.get("last_action_error", ""),
            "terminated": self.terminated,
            "truncated": self.truncated,
            "goal": raw.get("goal", ""),
            "chat_messages": [{"role": m["role"], "message": m["message"]} for m in raw.get("chat_messages", [])],
        }
        if self.render:
            from PIL import Image

            buffer = io.BytesIO()
            Image.fromarray(raw["screenshot"]).save(buffer, format="PNG")
            observation["images"] = [
                {
                    "name": "browser_screenshot",
                    "mime_type": "image/png",
                    "data": base64.b64encode(buffer.getvalue()).decode(),
                }
            ]
            for part in raw.get("goal_object", []):
                if part.get("type") == "image_url":
                    url = part["image_url"]
                    url = url["url"] if isinstance(url, dict) else url
                    if isinstance(url, str) and url.startswith("data:") and ";base64," in url:
                        prefix, encoded = url.split(";base64,", 1)
                        observation["images"].append(
                            {"name": "task_goal_image", "mime_type": prefix[5:], "data": encoded}
                        )
        return observation

    def reset(self, seed):
        with ExitStack() as stack:
            if self.memory_chat:
                stack.enter_context(patch("browsergym.core.env.Chat", MemoryChat))
            if self.cpu_headless:
                from browsergym.core import _get_global_playwright

                chromium = _get_global_playwright().chromium
                original_launch = chromium.launch

                def launch(**kwargs):
                    kwargs["headless"] = False
                    kwargs["args"] = list(kwargs.get("args", [])) + [
                        "--headless=new",
                        "--disable-gpu",
                        "--disable-software-rasterizer",
                        "--disable-dev-shm-usage",
                    ]
                    return original_launch(**kwargs)

                stack.enter_context(patch.object(chromium, "launch", side_effect=launch))
            self.raw, _ = self.env.reset(seed=seed)
        self.reward, self.terminated, self.truncated, self.steps = 0, False, False, 0
        return self._public()

    def observe(self):
        return self._public()

    def step(self, action):
        if self.terminated or self.truncated:
            raise InvalidAction("BrowserGym episode already ended")
        if not isinstance(action, dict):
            raise InvalidAction("browser action must be an object")
        op = action.get("op")
        fields = {
            "click": ["bid"],
            "fill": ["bid", "text"],
            "press": ["bid", "key"],
            "scroll": ["x", "y"],
            "mouse_click": ["x", "y"],
            "keyboard_type": ["text"],
            "send_msg_to_user": ["text"],
        }
        if self.navigation:
            fields.update(goto=["url"], go_back=[])
        if op not in fields or set(action) != {"op", *fields[op]}:
            raise InvalidAction("invalid browser operation or arguments")
        if op == "goto" and (
            not isinstance(action["url"], str) or urlparse(action["url"]).scheme not in {"http", "https"}
        ):
            raise InvalidAction("browser navigation requires an HTTP(S) URL")
        arguments = [action[field] for field in fields[op]]
        for name, value in zip(fields[op], arguments):
            if name in {"x", "y"}:
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise InvalidAction("browser coordinates must be finite numbers")
            elif not isinstance(value, str) or len(value) > 10000:
                raise InvalidAction("browser string arguments must be strings <= 10000 characters")
        code = f"{op}({', '.join(repr(value) for value in arguments)})"
        self.raw, self.reward, self.terminated, self.truncated, _ = self.env.step(code)
        self.steps += 1
        return self._public()

    def verify(self):
        if not self.terminated:
            self.reward, _, _, _ = self.env.unwrapped._task_validate()
        return RunResult(
            float(self.reward),
            {
                "success": float(self.reward >= 1),
                "browser_steps": float(self.steps),
                "terminated": float(self.terminated),
                "truncated": float(self.truncated),
            },
        )

    def close(self):
        if self.env is not None:
            self.env.close()
        if self.fixture is not None:
            self.fixture.close()
