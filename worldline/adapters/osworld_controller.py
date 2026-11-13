"""Run inside an independent OSWorld installation as an owned session provider."""

from __future__ import annotations

import argparse
import base64
import json
import math
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def desktop_action(action):
    """Only the trusted controller creates Python; no model code is executed."""
    op = action["op"]
    if op in {"click", "move"}:
        method = "click" if op == "click" else "moveTo"
        return f"pyautogui.{method}({int(action['x'])}, {int(action['y'])})"
    if op == "drag":
        return f"pyautogui.moveTo({int(action['x'])}, {int(action['y'])}); pyautogui.dragTo({int(action['to_x'])}, {int(action['to_y'])}, duration=0.5)"
    if op == "scroll":
        return f"pyautogui.scroll({int(action['amount'])})"
    if op == "type":
        return "pyautogui.write(" + repr(action["text"]) + ")"
    if op == "key":
        keys = action["key"].split("+")
        if len(keys) > 5 or any(not key or len(key) > 32 for key in keys):
            raise ValueError("invalid hotkey")
        return "pyautogui.hotkey(" + ", ".join(repr(key) for key in keys) + ")"
    if op == "wait":
        seconds = float(action["seconds"])
        if not 0 <= seconds <= 5:
            raise ValueError("invalid wait")
        return f"time.sleep({seconds})"
    raise ValueError("unsupported desktop action")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--provider", default="docker")
    parser.add_argument("--snapshot", required=True)
    args = parser.parse_args(argv)
    # Heavy OSWorld/provider dependencies live in the user's separate environment.
    from desktop_env.desktop_env import DesktopEnv
    from worldline.run import write_json

    request = json.loads(args.request.read_text())
    task_config = json.loads(args.task_config.read_text())
    import hashlib

    task_hash = hashlib.sha256(args.task_config.read_bytes()).hexdigest()
    if request.get("task_config_sha256") and request["task_config_sha256"] != task_hash:
        raise ValueError("OSWorld task config digest mismatch")
    session_id = request["session_id"]
    env = DesktopEnv(path_to_vm=args.snapshot, provider_name=args.provider, action_space="pyautogui", headless=True)
    try:
        observation = env.reset(task_config=task_config, seed=request["seed"])
        mutex = threading.Lock()

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                nonlocal observation
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= 1024 * 1024:
                        raise ValueError("invalid request size")
                    body = json.loads(self.rfile.read(size))
                    if body["session_id"] != session_id:
                        raise ValueError("session identity mismatch")
                    with mutex:
                        operation = self.path.removeprefix("/worldline/")
                        if operation == "act":
                            observation, _, _, _ = env.step(desktop_action(body["params"]))
                            result = {}
                        elif operation == "observe":
                            screenshot = observation["screenshot"]
                            import io
                            from PIL import Image

                            with Image.open(io.BytesIO(screenshot)) as image:
                                width, height = image.size
                            result = {
                                "screenshot_png_base64": base64.b64encode(screenshot).decode(),
                                "width": width,
                                "height": height,
                            }
                        elif operation == "verify":
                            score = float(env.evaluate())
                            if not math.isfinite(score):
                                raise ValueError("non-finite OSWorld score")
                            result = {"verified": True, "success": score >= 1, "native_score": score}
                        else:
                            raise ValueError("unsupported controller operation")
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(json.dumps({"session_id": session_id, "result": result}).encode())
                except Exception:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(
                        json.dumps({"session_id": session_id, "error": "controller operation failed"}).encode()
                    )

            def log_message(self, *_):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        write_json(
            args.ready,
            {
                "schema_version": "worldline.session/0.1",
                **request,
                "session_id": session_id,
                "endpoint": f"http://127.0.0.1:{server.server_port}",
                "task_config_sha256": task_hash,
            },
        )
        try:
            for line in sys.stdin:
                if json.loads(line).get("method") == "close":
                    break
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
