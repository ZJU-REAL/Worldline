from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from worldline.adapters.desktop import DesktopEnvironment
from worldline.adapters.osworld_controller import desktop_action
from worldline.contracts import Task
from worldline.errors import InvalidAction


class DesktopAdapterTest(unittest.TestCase):
    def test_structured_actions_feedback_and_trusted_verdict(self):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append((self.path, body))
                result = {
                    "screenshot_png_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgYGAAAAAEAAH2FzhVAAAAAElFTkSuQmCC",
                    "width": 1,
                    "height": 1,
                }
                if self.path.endswith("verify"):
                    result = {"verified": True, "success": len(requests) > 2, "native_score": 1}
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({"session_id": body["session_id"], "result": result}).encode())

            def log_message(self, *_):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        adapter = None
        try:
            with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"WORLDLINE_STATE_DIR": temp}):
                manifest = Path(temp) / "session.json"
                manifest.write_text(
                    json.dumps(
                        {
                            "schema_version": "worldline.session/0.1",
                            "session_id": "desktop",
                            "task_id": "task",
                            "task_version": "v1",
                            "seed": 0,
                            "source_revision": "source",
                            "endpoint": f"http://127.0.0.1:{server.server_port}",
                        }
                    )
                )
                adapter = DesktopEnvironment(
                    Task("task", "x", "cua.desktop", "v1"),
                    {"source_revision": "source", "session_manifest": str(manifest)},
                )
                adapter.step({"op": "click", "x": 0, "y": 0})
                self.assertEqual(adapter.verify().reward, 1)
                count = len(requests)
                with self.assertRaises(InvalidAction):
                    adapter.step({"op": "python", "code": "anything"})
                self.assertEqual(len(requests), count)
                adapter.close()
                adapter = None
        finally:
            if adapter:
                adapter.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_controller_quotes_text_instead_of_executing_it(self):
        text = "'); __import__('os').system('unexpected'); #"
        code = desktop_action({"op": "type", "text": text})
        calls = []

        class GUI:
            def write(self, value):
                calls.append(value)

        exec(code, {"pyautogui": GUI()})
        self.assertEqual(calls, [text])
