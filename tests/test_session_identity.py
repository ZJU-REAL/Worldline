import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from worldline.contracts import Task
from worldline.errors import EnvironmentError
from worldline.runtime.provisioning import ExternalSession


class SessionIdentityTest(unittest.TestCase):
    def test_descriptor_cannot_disable_checks_or_use_bool_as_seed(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"WORLDLINE_STATE_DIR": temp}):
            root = Path(temp)
            path = root / "descriptor.json"
            descriptor = {
                "schema_version": "worldline.session/0.1",
                "task_id": "x",
                "task_version": "v1",
                "seed": 1,
                "source_revision": "src",
                "session_id": "one",
                "socket_path": str(root / "sock"),
                "identity_verified": False,
            }
            config = {"session_manifest": str(path), "source_revision": "src"}
            task = Task("x", "x", "rua.socket", "v1", seed=1)
            path.write_text(json.dumps(descriptor))
            with self.assertRaisesRegex(EnvironmentError, "cannot disable"):
                ExternalSession(task, config)
            descriptor.pop("identity_verified")
            descriptor["seed"] = True
            path.write_text(json.dumps(descriptor))
            with self.assertRaisesRegex(EnvironmentError, "seed mismatch"):
                ExternalSession(task, config)
