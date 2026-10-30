from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from worldline.eval import evaluate_harbor, harbor_command


def make_task(root: Path) -> Path:
    task_dir = root / "task"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text(
        """
[task]
name = "example/task"

[metadata]
split = "test"

[metadata.seeds]
default = 7
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (task_dir / "instruction.md").write_text("Do the task.\n", encoding="utf-8")
    (task_dir / "DIGEST").write_text("sha256:example\n", encoding="utf-8")
    return task_dir


class HarborEvaluationTest(unittest.TestCase):
    def test_builds_command(self) -> None:
        command = harbor_command(
            Path("task"),
            "oracle",
            "model",
            extra_args=("--attempts", "2"),
            jobs_dir=Path("runs"),
            job_name="job",
        )
        self.assertEqual(
            command,
            (
                "harbor",
                "run",
                "-y",
                "-p",
                "task",
                "-a",
                "oracle",
                "-m",
                "model",
                "--jobs-dir",
                "runs",
                "--job-name",
                "job",
                "--attempts",
                "2",
            ),
        )

    def test_dry_run_writes_record_and_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            record, run_dir = evaluate_harbor(
                make_task(root), agent="oracle", runs_dir=root / "runs", dry_run=True
            )

            saved = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            events = [json.loads(line) for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(record.status, "planned")
            self.assertEqual(saved["task_version"], "sha256:example")
            self.assertEqual(saved["metadata"]["seed"], 7)
            self.assertEqual([event["kind"] for event in events], ["run.created", "run.planned"])

    def test_executes_backend_and_captures_log(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            backend = root / "fake-harbor"
            backend.write_text("#!/bin/sh\necho fake backend\nexit 0\n", encoding="utf-8")
            backend.chmod(0o755)

            record, run_dir = evaluate_harbor(
                make_task(root), agent="oracle", runs_dir=root / "runs", executable=str(backend)
            )

            self.assertEqual(record.status, "completed")
            self.assertEqual(record.exit_code, 0)
            self.assertEqual((run_dir / "backend.log").read_text(encoding="utf-8"), "fake backend\n")

    def test_automatically_ingests_backend_trial(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            backend = root / "fake-harbor"
            backend.write_text(
                """#!/usr/bin/env python3
import json, pathlib, sys
args = sys.argv[1:]
jobs = pathlib.Path(args[args.index('--jobs-dir') + 1])
name = args[args.index('--job-name') + 1]
trial = jobs / name / 'example__trial'
(trial / 'verifier').mkdir(parents=True)
(trial / 'result.json').write_text(json.dumps({'id': 'trial-3', 'task_name': 'example/task'}))
(trial / 'verifier/reward.json').write_text(json.dumps({'success': 1, 'progress': 1}))
(trial / 'verifier/reward.txt').write_text('1\\n')
print('fake backend with result')
""",
                encoding="utf-8",
            )
            backend.chmod(0o755)

            _, run_dir = evaluate_harbor(
                make_task(root), agent="oracle", runs_dir=root / "runs", executable=str(backend)
            )
            result = json.loads((run_dir / "result.normalized.json").read_text(encoding="utf-8"))
            collection = json.loads((run_dir / "results.normalized.json").read_text(encoding="utf-8"))

            self.assertEqual(result["outcome"], "success")
            self.assertEqual(result["trial_id"], "trial-3")
            self.assertEqual(len(collection["results"]), 1)


if __name__ == "__main__":
    unittest.main()
