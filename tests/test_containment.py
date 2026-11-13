from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from worldline.runtime.containment import worker_command
from worldline.runtime.container_guard import process_identity
from worldline.tasks import load_native_task


class ContainmentTest(unittest.TestCase):
    def test_pinned_container_has_limits_no_capabilities_and_no_inherited_credentials(self):
        task = Path(__file__).resolve().parents[1] / "examples/rua/reach-train"
        original = load_native_task(task)
        definition = replace(
            original,
            task=replace(
                original.task,
                metadata={
                    **original.task.metadata,
                    "isolation": {
                        "backend": "docker",
                        "image": "image@sha256:" + "0" * 64,
                        "memory_mb": 1024,
                        "pids_limit": 64,
                    },
                },
            ),
        )
        with (
            tempfile.TemporaryDirectory() as temp,
            patch("worldline.runtime.containment.shutil.which", return_value="docker"),
        ):
            command, name = worker_command(definition, Path(temp), "python")
            self.assertIn("1024m", command)
            self.assertIn("no-new-privileges", command)
            self.assertIn("none", command)
            self.assertNotIn("WORLDLINE_API_KEY", str(command))
            self.assertTrue(name.startswith("worldline-"))
        import os

        self.assertTrue(process_identity(os.getpid()))
        self.assertIsNone(process_identity(999999999))
