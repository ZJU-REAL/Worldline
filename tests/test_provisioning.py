from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from worldline.contracts import Task
from worldline.errors import EnvironmentError
from worldline.runtime.provisioning import ExternalSession


class ProvisioningTest(unittest.TestCase):
    def descriptor(self, root):
        return {
            "schema_version": "worldline.session/0.1",
            "session_id": "unique-session",
            "task_id": "task",
            "task_version": "v1",
            "seed": 7,
            "source_revision": "source",
            "socket_path": str(root / "sim.sock"),
            "image_root": str(root),
        }

    def test_borrowed_sessions_are_exclusive_single_use_and_identity_checked(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"WORLDLINE_STATE_DIR": temp}):
            root = Path(temp)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(self.descriptor(root)))
            task = Task("task", "x", "rua.socket", "v1", seed=7)
            config = {"session_manifest": str(manifest), "source_revision": "source"}
            session = ExternalSession(task, config)
            try:
                with self.assertRaisesRegex(EnvironmentError, "already owned"):
                    ExternalSession(task, config)
            finally:
                session.close()
            with self.assertRaisesRegex(EnvironmentError, "already been used"):
                ExternalSession(task, config)
            with self.assertRaisesRegex(EnvironmentError, "seed mismatch"):
                ExternalSession(Task("task", "x", "rua.socket", "v1", seed=8), config)

    def test_owned_provider_receives_identity_and_is_reaped_on_close(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"WORLDLINE_STATE_DIR": temp}):
            root = Path(temp)
            script = root / "provider.py"
            script.write_text(
                "import json,sys\nfrom pathlib import Path\nr=json.loads(Path(sys.argv[1]).read_text());r.update(socket_path=str(Path(sys.argv[2]).parent/'sim.sock'),image_root=str(Path(sys.argv[2]).parent));Path(sys.argv[2]).write_text(json.dumps(r));sys.stdin.readline()\n"
            )
            task = Task("task", "x", "rua.socket", "v1", seed=7)
            session = ExternalSession(
                task,
                {
                    "source_revision": "source",
                    "provider": {
                        "command": [sys.executable, str(script), "{request_file}", "{ready_file}"],
                        "startup_timeout_sec": 2,
                    },
                },
            )
            process = session.process
            self.assertTrue(session.verified)
            self.assertIsNone(process.poll())
            session.close()
            self.assertIsNotNone(process.poll())
            recorded = json.loads((session.directory / "request.json").read_text())
            self.assertEqual(recorded["seed"], 7)

    def test_invalid_identity_closes_the_provider(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"WORLDLINE_STATE_DIR": temp}):
            root = Path(temp)
            manifest = root / "wrong.json"
            descriptor = self.descriptor(root)
            descriptor["task_version"] = "wrong"
            manifest.write_text(json.dumps(descriptor))
            with self.assertRaisesRegex(EnvironmentError, "task_version mismatch"):
                ExternalSession(
                    Task("task", "x", "rua.socket", "v1", seed=7),
                    {"session_manifest": str(manifest), "source_revision": "source"},
                )
