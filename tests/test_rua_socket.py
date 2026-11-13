from __future__ import annotations

import hashlib
import json
import socketserver
import tempfile
import threading
import unittest
from pathlib import Path

from worldline.adapters.rua.socket import RUASocketEnvironment
from worldline.contracts import Task
from worldline.errors import EnvironmentError, InvalidAction


class SocketAdapterTest(unittest.TestCase):
    def test_matches_existing_rua_wire_protocol_and_checks_camera_integrity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "frame.png"
            image.write_bytes(b"camera-bytes")
            requests = []
            observation = {
                "images": {"front": {"path": str(image), "sha256": hashlib.sha256(image.read_bytes()).hexdigest()}},
                "state": {"robot_eef_pos": [0, 0, 0]},
                "simulation_steps": 1,
                "action_schema": {"low": [-1], "high": [1]},
                "simulation_paused": True,
                "private_state_path": "/private/answer",
            }

            class Handler(socketserver.StreamRequestHandler):
                def handle(self):
                    request = json.loads(self.rfile.readline())
                    requests.append(request)
                    result = (
                        {"simulation_verified": True, "success": True, "simulation_steps": 1}
                        if request["method"] == "evaluate"
                        else observation
                    )
                    self.wfile.write(json.dumps({"id": request["id"], "result": result}).encode() + b"\n")

            server = socketserver.UnixStreamServer(str(root / "sim.sock"), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                adapter = RUASocketEnvironment(
                    Task("x", "reach", "rua.socket", "1"),
                    {
                        "socket_path": str(root / "sim.sock"),
                        "image_root": str(root),
                        "source_revision": "fixed",
                        "allow_unverified_session": True,
                    },
                )
                self.assertNotIn("private_state_path", adapter.initial)
                self.assertNotIn("path", adapter.initial["images"][0])
                adapter.step({"actions": [[0.1]]})
                self.assertEqual(requests[-1]["method"], "step")
                self.assertEqual(requests[-1]["params"], {"actions": [[0.1]]})
                self.assertEqual(adapter.verify().reward, 1)
                with self.assertRaises(InvalidAction):
                    adapter.reset(0)
                image.write_bytes(b"changed")
                with self.assertRaisesRegex(EnvironmentError, "hash mismatch"):
                    adapter.observe()
            finally:
                if "adapter" in locals():
                    adapter.close()
                server.shutdown()
                server.server_close()
                thread.join()
