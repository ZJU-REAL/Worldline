from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from worldline.contracts import Task
from worldline.tasks import NativeTask
from worldline.runtime.process import ProcessEnvironment


class WorkerMediaTest(unittest.TestCase):
    def test_large_binary_observation_uses_shared_artifacts_instead_of_pipe_payload(self):
        script = """import base64,types
import worldline.runtime.worker as w
class Environment:
 def __init__(self,*_):self.data=base64.b64encode(b'x'*(13*1024*1024)).decode()
 def spec(self):return {'action_schema':{'type':'object'}}
 def observe(self):return {'blob':{'mime_type':'application/octet-stream','data':self.data}}
 def close(self):pass
w.get_environment_plugin=lambda _:types.SimpleNamespace(version='v1',factory=Environment)
w.main()
"""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            definition = NativeTask(
                Task("test", "test", "test", "v1", metadata={"environment_version": "v1"}), {}, 30, 1, 1, 0
            )
            original = subprocess.Popen
            with patch(
                "worldline.runtime.process.subprocess.Popen",
                side_effect=lambda _, **kwargs: original([sys.executable, "-c", script], **kwargs),
            ):
                environment = ProcessEnvironment(definition, root)
            try:
                result = environment.start()
                self.assertGreater(len(result["blob"]["data"]), 16 * 1024 * 1024)
                blobs = list((root / "media").glob("*.bin"))
                self.assertEqual(len(blobs), 1)
                self.assertEqual(blobs[0].stat().st_size, 13 * 1024 * 1024)
                self.assertEqual(environment.pending, b"")
            finally:
                environment.close()
