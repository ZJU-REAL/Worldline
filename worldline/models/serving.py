"""One owned model service per evaluation, shared by bounded trial workers."""

import json
import base64
import os
import shutil
import socket
import subprocess
import sys
import time
import struct
import zlib
import urllib.request
from contextlib import AbstractContextManager
from importlib.util import find_spec
from pathlib import Path

from worldline.errors import HTTPRequestError
from worldline.models.config import ModelConfig, ResolvedModel
from worldline.runtime.http import http_json


def request_json(url, *, api_key_env, body=None, timeout=15):
    headers = {"Content-Type": "application/json"}
    if key := os.environ.get(api_key_env):
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None, headers=headers)
    return http_json(request, time.monotonic() + timeout, 2 * 1024 * 1024)


def check_server(config):
    backend = config.effective_backend
    if backend == "vllm" and not config.server_executable and find_spec("vllm") is None:
        raise ValueError(
            "vLLM is absent; install worldline-agent[serving] in a GPU serving environment, "
            "select --server-executable /path/to/vllm, or supply --base-url"
        )
    executable = config.server_executable
    if backend == "llama-cpp":
        executable = executable or "llama-server"
    if backend == "command":
        executable = config.server_command[0]
    if executable and not shutil.which(executable):
        raise ValueError(f"model server executable unavailable: {executable}")


def server_argv(config: ModelConfig, name, port):
    ref = (
        str(Path(config.reference).expanduser().resolve())
        if Path(config.reference).expanduser().exists()
        else config.reference
    )
    if config.effective_backend == "vllm":
        head = (
            [config.server_executable, "serve"]
            if config.server_executable
            else [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model"]
        )
        argv = [*head, ref, "--host", "127.0.0.1", "--port", str(port), "--served-model-name", name]
        if config.revision:
            argv += ["--revision", config.revision]
    elif config.effective_backend == "llama-cpp":
        argv = [
            config.server_executable or "llama-server",
            "--model",
            ref,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--alias",
            name,
            "--parallel",
            "1",
            "--ctx-size",
            "16384",
        ]
    else:
        replacements = {"model": ref, "name": name, "host": "127.0.0.1", "port": str(port)}
        argv = list(config.server_command)
        for index, item in enumerate(argv):
            for key, value in replacements.items():
                item = item.replace("{" + key + "}", value)
            argv[index] = item
    return [*argv, *config.server_args]


class ModelService(AbstractContextManager):
    def __init__(self, config, directory, *, settings=None):
        self.config, self.directory = config, Path(directory)
        self.settings = settings if settings is not None else config.identity()
        self.process = self.log = None
        self.command = None

    def __enter__(self):
        config = self.config
        name = config.name or config.reference
        try:
            if config.effective_backend == "endpoint":
                base_url = config.base_url.rstrip("/")
            else:
                check_server(config)
                with socket.socket() as listener:
                    listener.bind(("127.0.0.1", 0))
                    port = listener.getsockname()[1]
                base_url = f"http://127.0.0.1:{port}/v1"
                self.command = server_argv(config, name, port)
                self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
                log_path = self.directory / "server.log"
                fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                self.log = os.fdopen(fd, "wb")
                self.process = subprocess.Popen(
                    [sys.executable, "-m", "worldline.runtime.command_guard"],
                    stdin=subprocess.PIPE,
                    stdout=self.log,
                    stderr=self.log,
                    start_new_session=True,
                )
                self.process.stdin.write(json.dumps(self.command).encode() + b"\n")
                self.process.stdin.flush()
            self.model = ResolvedModel(
                name, base_url, config.api_key_env, config.revision, config.protocol, config.vision, self.settings
            )
            self.readiness = self._ready()
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def _ready(self):
        deadline = time.monotonic() + (self.config.startup_timeout_sec if self.process else 20)
        while True:
            if time.monotonic() >= deadline:
                raise ValueError(f"model readiness timed out; inspect {self.directory / 'server.log'}")
            if self.process and self.process.poll() is not None:
                raise ValueError(f"model server exited before readiness; inspect {self.directory / 'server.log'}")
            try:
                response = request_json(
                    self.model.base_url + "/models",
                    api_key_env=self.model.api_key_env,
                    timeout=min(5, max(0.1, deadline - time.monotonic())),
                )
                entries = response.get("data") if isinstance(response, dict) else None
                if not isinstance(entries, list):
                    raise ValueError("model endpoint /models must return a data list")
                names = [entry.get("id") for entry in entries if isinstance(entry, dict)]
                if self.model.name not in names:
                    if self.process:
                        time.sleep(0.2)
                        continue
                    raise ValueError(f"model {self.model.name!r} is not served; available models: {names[:20]}")
                if self.process and self.process.poll() is not None:
                    raise ValueError("owned server exited during readiness")
                return {"models": names, "protocol": self.config.protocol}
            except HTTPRequestError as error:
                # Some authenticated providers omit /models. A completion probe
                # still verifies their actual model/protocol before any task starts.
                if not self.process and error.status in {404, 405}:
                    return {"models": None, "protocol": self.config.protocol}
                if not self.process or error.status in {400, 401, 403, 422} or time.monotonic() >= deadline:
                    raise ValueError(
                        f"model readiness failed ({error}); check URL and {self.config.api_key_env}"
                    ) from error
                time.sleep(min(0.2, max(0, deadline - time.monotonic())))

    def __exit__(self, *_):
        if self.process:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=5)
        if self.log:
            self.log.close()
        return False


def probe_model(model, *, tools=False):
    """Check protocol acceptance, not agent quality. This is not a scored trial."""

    def chunk(kind, data):
        return struct.pack("!I", len(data)) + kind + data + struct.pack("!I", zlib.crc32(kind + data))

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack("!2I5B", 32, 32, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress((b"\0" + b"\xff" * 96) * 32)) + chunk(b"IEND", b"")
    image_url = "data:image/png;base64," + base64.b64encode(png).decode()
    if model.protocol == "responses":
        body = {"model": model.name, "input": "Reply OK.", "max_output_tokens": 64, "stream": False}
        if model.vision:
            body["input"] = [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Reply OK."},
                        {"type": "input_image", "image_url": image_url},
                    ],
                }
            ]
        suffix = "/responses"
    else:
        content = [{"type": "text", "text": "Reply OK."}]
        if model.vision:
            # A valid RGB PNG verifies image transport, not visual intelligence.
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                }
            )
        body = {
            "model": model.name,
            "messages": [{"role": "user", "content": content if model.vision else "Reply OK."}],
            "max_tokens": 64,
            "stream": False,
        }
        if tools:
            body.update(
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "observe",
                            "description": "Read observation",
                            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                        },
                    }
                ],
                tool_choice="auto",
            )
        suffix = "/chat/completions"
    try:
        response = request_json(model.base_url + suffix, api_key_env=model.api_key_env, body=body, timeout=120)
        if model.protocol == "chat":
            message = response["choices"][0]["message"]
            if not isinstance(message, dict):
                raise ValueError("no chat message")
        elif not isinstance(response.get("output"), list):
            raise ValueError("no Responses output")
    except (HTTPRequestError, ValueError, KeyError, IndexError, TypeError) as error:
        hint = "; try --harness json if function calling is unsupported" if tools else ""
        raise ValueError(f"model protocol probe failed: {error}{hint}") from error
    return {
        "protocol": model.protocol,
        "tools_requested": tools,
        "vision_requested": model.vision,
        "usage": response.get("usage"),
        "scored": False,
    }
