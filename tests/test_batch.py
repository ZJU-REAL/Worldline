from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from worldline.batch import evaluate_many_harbor


def task(root: Path, name: str) -> Path:
    directory = root / name
    directory.mkdir()
    (directory / "task.toml").write_text(f'[task]\nname = "example/{name}"\n', encoding="utf-8")
    (directory / "instruction.md").write_text("Do it.\n", encoding="utf-8")
    return directory


class BatchEvaluationTest(unittest.TestCase):
    def test_plans_multiple_tasks_and_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runs = root / "runs"
            manifest = evaluate_many_harbor(
                [task(root, "one"), task(root, "two")],
                agent="oracle",
                runs_dir=runs,
                dry_run=True,
                max_workers=2,
            )

            self.assertEqual(manifest["tasks"], 2)
            self.assertEqual(manifest["statuses"], {"planned": 2})
            saved = json.loads((runs / "batch.json").read_text(encoding="utf-8"))
            self.assertEqual(len(saved["runs"]), 2)
            self.assertTrue(all(row["run_dir"] for row in saved["runs"]))

    def test_records_invalid_task_without_stopping_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = evaluate_many_harbor(
                [task(root, "valid"), root / "missing"],
                agent="oracle",
                runs_dir=root / "runs",
                dry_run=True,
            )
            self.assertEqual(manifest["statuses"], {"error": 1, "planned": 1})


if __name__ == "__main__":
    unittest.main()
