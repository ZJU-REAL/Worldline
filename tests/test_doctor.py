from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from worldline.doctor import doctor


class DoctorTest(unittest.TestCase):
    def test_reports_missing_runtime_without_hiding_framework_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "tasks").mkdir()
            (root / "worldline").mkdir()
            with patch("worldline.doctor.shutil.which", side_effect=lambda name: "/bin/uv" if name == "uv" else None):
                with patch("worldline.doctor._version", return_value="uv test"):
                    result = doctor(root)

            self.assertTrue(result["framework_ready"])
            self.assertFalse(result["evaluation_ready"])
            self.assertFalse(result["checks"]["harbor"]["ok"])
            self.assertFalse(result["checks"]["container"]["ok"])
            self.assertEqual(len(result["recommendations"]), 3)


if __name__ == "__main__":
    unittest.main()
