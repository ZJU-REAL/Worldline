from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from worldline.report import build_report, write_report


def result(root: Path, name: str, task: str, outcome: str, reward: float | None, tokens: int = 0) -> None:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "result.normalized.json").write_text(
        json.dumps(
            {
                "task_id": task,
                "outcome": outcome,
                "reward": reward,
                "usage": {"n_input_tokens": tokens},
                "metrics": {"sim_steps": 10},
            }
        ),
        encoding="utf-8",
    )


class ReportTest(unittest.TestCase):
    def test_excludes_infrastructure_errors_from_success_rate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result(root, "a", "task-a", "success", 1, 100)
            result(root, "b", "task-a", "failure", 0, 50)
            result(root, "c", "task-b", "infra_error", None)

            report = build_report([root])

            self.assertEqual(report["summary"]["trials"], 3)
            self.assertEqual(report["summary"]["counted_trials"], 2)
            self.assertEqual(report["summary"]["success_rate"], 0.5)
            self.assertEqual(report["summary"]["infra_error_rate"], 1 / 3)
            self.assertEqual(report["summary"]["usage_totals"]["n_input_tokens"], 150)
            self.assertEqual(report["tasks"]["task-a"]["success_rate"], 0.5)

    def test_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result(root, "a", "task-a", "success", 1)
            output = root / "report.json"
            write_report([root], output)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["schema_version"], "worldline.report/0.1")


if __name__ == "__main__":
    unittest.main()
