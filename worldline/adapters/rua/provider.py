"""Provision a fresh approved rua-robot/1.0 simulator command for each trial."""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

from worldline.run import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--command-file", type=Path, required=True)
    parser.add_argument("--startup-timeout", type=float, default=60)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    command_raw = args.command_file.read_bytes()
    command = json.loads(command_raw)
    if not isinstance(command, list) or not command or not all(isinstance(part, str) for part in command):
        raise ValueError("simulator command must be a JSON argument array")
    digest = "sha256:" + hashlib.sha256(command_raw).hexdigest()
    if request.get("controller_revision") and request["controller_revision"] != digest:
        raise ValueError("simulator command digest mismatch")
    with tempfile.TemporaryDirectory(prefix="worldline-rua-") as temp:
        socket_path = str(Path(temp) / "sim.sock")
        images = args.ready.parent / "simulator-images"
        images.mkdir(parents=True, exist_ok=True)
        fields = {**request, "socket_path": socket_path, "image_root": str(images)}
        process = subprocess.Popen(
            [part.format_map(fields) for part in command],
            stdin=subprocess.DEVNULL,
            stdout=sys.stderr,
            stderr=sys.stderr,
        )
        try:
            deadline = time.monotonic() + args.startup_timeout
            while True:
                if process.poll() is not None:
                    raise RuntimeError("simulator exited before readiness")
                if time.monotonic() >= deadline:
                    raise TimeoutError("simulator readiness timeout")
                if Path(socket_path).exists():
                    try:
                        ident = uuid.uuid4().hex
                        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                            connection.settimeout(min(2, max(0.01, deadline - time.monotonic())))
                            connection.connect(socket_path)
                            connection.sendall(
                                json.dumps({"id": ident, "method": "observe", "params": {}}).encode() + b"\n"
                            )
                            with connection.makefile("rb") as stream:
                                response = json.loads(stream.readline(8 * 1024 * 1024 + 1))
                        if (
                            response.get("id") == ident
                            and "result" in response
                            and "action_schema" in response["result"]
                        ):
                            break
                    except (OSError, ValueError):
                        pass
                time.sleep(0.05)
            write_json(
                args.ready,
                {
                    "schema_version": "worldline.session/0.1",
                    **request,
                    "socket_path": socket_path,
                    "image_root": str(images),
                    "controller_revision": digest,
                },
            )
            for line in sys.stdin:
                if json.loads(line).get("method") == "close":
                    break
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=0.2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
