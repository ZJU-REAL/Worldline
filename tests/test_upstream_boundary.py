from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from worldline.adapters.efe import resolve_checkout, launch
from worldline.doctor import doctor


def checkout(root):
    (root / "tasks").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "docs/core_set.toml").write_text("")
    (root / "scripts").mkdir()
    (root / "scripts/task_conventions.py").write_text("")


class UpstreamBoundaryTest(unittest.TestCase):
    def test_submodule_and_external_checkout_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            vendored, external = root / "repo/third_party/embodied-first-exam", root / "external"
            checkout(vendored)
            checkout(external)
            self.assertEqual(resolve_checkout(start=root / "repo", environ={}), vendored)
            self.assertEqual(resolve_checkout(start=root, environ={"WORLDLINE_EFE_ROOT": str(external)}), external)
            with self.assertRaisesRegex(ValueError, "not an EFE checkout"):
                resolve_checkout(root / "missing", start=root / "repo", environ={})
            with self.assertRaisesRegex(ValueError, "checkout unavailable"):
                resolve_checkout(start=root / "other", environ={})

    def test_launcher_uses_explicit_child_process_and_no_import_hook(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkout(root)
            with patch("worldline.adapters.efe.subprocess.call", return_value=0) as call:
                self.assertEqual(launch(["run", "--list"], root), 0)
            self.assertEqual(call.call_args.args[0][1:], [str(root / "scripts/efe.py"), "run", "--list"])
            self.assertEqual(call.call_args.kwargs["env"]["EFE_ROOT"], str(root))
            with patch("worldline.adapters.efe.subprocess.call", return_value=0) as external:
                launch(["run", "--list"], root, python="/external/efe/bin/python")
            self.assertEqual(external.call_args.args[0][0], "/external/efe/bin/python")

    def test_installed_core_does_not_require_uv_repository_or_efe(self):
        with tempfile.TemporaryDirectory() as temp, patch("worldline.doctor.shutil.which", return_value=None):
            result = doctor(Path(temp))
        self.assertTrue(result["framework_ready"])
        self.assertFalse(result["checks"]["repository"]["ok"])
        self.assertFalse(result["evaluation_ready"])

    def test_installed_container_binary_without_daemon_is_not_ready(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch("worldline.doctor.shutil.which", return_value="/fake/bin"):
                with patch("worldline.doctor._probe", return_value={"ok": False, "detail": "connection refused"}):
                    with patch("worldline.doctor._version", return_value="version"):
                        result = doctor(Path(temp))
        self.assertFalse(result["evaluation_ready"])
