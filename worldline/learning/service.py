"""Episode service owned by the trainer deployment; environments stay stateful."""

from __future__ import annotations

import argparse
import json
import os
import tomllib
import urllib.request
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from worldline.learning.bridge import EpisodeRegistry
from worldline.runtime.shared_resources import HostResourcePool


class EpisodeClient:
    def __init__(self, endpoint, api_key_env="WORLDLINE_TRAINING_KEY"):
        from urllib.parse import urlparse

        parsed = urlparse(endpoint)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("training endpoint must be a plain HTTP(S) URL")
        self.endpoint, self.key_env = endpoint.rstrip("/"), api_key_env

    def call(self, operation, *, timeout_sec=60, **payload):
        from worldline.runtime.http import http_json

        headers = {"Content-Type": "application/json"}
        if os.environ.get(self.key_env):
            headers["Authorization"] = "Bearer " + os.environ[self.key_env]
        request = urllib.request.Request(
            self.endpoint + "/" + operation, data=json.dumps(payload, allow_nan=False).encode(), headers=headers
        )
        return http_json(request, time.monotonic() + timeout_sec, 32 * 1024 * 1024)


def server_for(registry, host, port, token=None):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if token and self.headers.get("Authorization") != "Bearer " + token:
                self.send_error(401)
                return
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 16 * 1024 * 1024:
                    raise ValueError("invalid training request size")
                payload = json.loads(self.rfile.read(size))
                operation = self.path.removeprefix("/")
                method = {
                    "begin": registry.begin,
                    "advance": registry.advance,
                    "context": registry.context,
                    "finish": registry.finish,
                    "abort": registry.abort,
                    "release_tool": registry.release_tool,
                }[operation]
                result = method(**payload)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(result, allow_nan=False).encode())
            except (KeyError, ValueError, RuntimeError, OSError):
                self.send_error(409, "training episode request failed; discard unscorable rollout")

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--api-key-env", default="WORLDLINE_TRAINING_KEY")
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    token = os.environ.get(args.api_key_env)
    if args.host not in {"127.0.0.1", "localhost", "::1"} and not token:
        parser.error("remote training service requires an API key and deployment TLS termination")
    tasks = {key: (args.config.parent / value).resolve() for key, value in config["tasks"].items()}
    pool = HostResourcePool(config.get("cpus", os.cpu_count() or 1), tuple(config.get("gpu_devices", ())))
    registry = EpisodeRegistry(
        tasks,
        runs_dir=(args.config.parent / config.get("runs_dir", "runs")).resolve(),
        resources=pool,
        max_sessions=config.get("max_sessions", 64),
        session_options=config.get("session", {}),
    )
    server = server_for(registry, args.host, args.port, token)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        registry.close()


if __name__ == "__main__":
    main()
