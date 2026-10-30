from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.test_datasets import make_result
from worldline.distill import build_distillation_bundle


class DistillationBundleTest(unittest.TestCase):
    def test_builds_report_and_all_training_views(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            make_result(root, "success", "task-a", "success", 1.0)
            make_result(root, "failure", "task-a", "failure", 0.0)

            manifest = build_distillation_bundle([root / "runs"], root / "bundle")

            self.assertEqual(manifest["report"]["summary"]["trials"], 2)
            self.assertEqual(manifest["datasets"]["sft"]["export"]["records"], 1)
            self.assertEqual(manifest["datasets"]["preference"]["export"]["records"], 1)
            self.assertEqual(manifest["datasets"]["rl"]["export"]["records"], 2)
            self.assertTrue((root / "bundle" / "manifest.json").is_file())
            for format_name in ("sft", "preference", "rl"):
                self.assertTrue(manifest["datasets"][format_name]["validation"]["valid"])


if __name__ == "__main__":
    unittest.main()
