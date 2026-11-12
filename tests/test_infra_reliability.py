from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from worldline.report import build_report
from worldline.run import RunRecord, write_json, write_record
from worldline.runtime.batch import evaluate_many_native, ResourcePool

TASK = Path(__file__).resolve().parents[1] / "examples/rua/reach-train"


class BatchReliabilityTest(unittest.TestCase):
    def fake_backend(self, root, outcomes):
        calls = []

        def run(path, **kwargs):
            index = len(calls)
            calls.append(kwargs)
            outcome, retryable = outcomes[index]
            directory = root / f"attempt-{index}"
            directory.mkdir()
            record = RunRecord(
                str(index),
                "task",
                "v1",
                "native",
                "test",
                None,
                (),
                "infra_error" if outcome == "infra_error" else "completed",
            )
            write_record(directory, record)
            kwargs["on_created"](record, directory)
            write_json(
                directory / "result.normalized.json",
                {
                    "trial_id": str(index),
                    "task_id": "task",
                    "source": str(directory),
                    "outcome": outcome,
                    "reward": 1 if outcome == "success" else 0,
                    "retryable": retryable,
                    "usage": {"model_requests": 2},
                    "logical_trial_id": kwargs["logical_trial_id"],
                },
            )
            return record, directory

        return calls, run

    def test_fresh_retry_is_preserved_and_only_final_attempt_is_counted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            calls, run = self.fake_backend(root, [("infra_error", True), ("success", False)])
            with patch("worldline.runtime.batch.evaluate_native", side_effect=run):
                batch = evaluate_many_native(
                    [TASK], runs_dir=root, batch_id="test", agent_factory=None, agent_name="test", retry_delay_sec=0
                )
            self.assertEqual(len(calls), 2)
            self.assertEqual(len(batch["runs"][0]["attempts"]), 2)
            self.assertEqual(calls[0]["logical_trial_id"], calls[1]["logical_trial_id"])
            self.assertFalse(json.loads((root / "attempt-0/result.normalized.json").read_text())["counted"])
            report = build_report([root])
            self.assertEqual(report["summary"]["trials"], 1)
            self.assertEqual(report["superseded_attempts"], 1)
            self.assertEqual(report["all_attempt_usage_totals"]["model_requests"], 4)

    def test_failures_and_permanent_errors_do_not_get_free_attempts(self):
        for outcome in ("failure", "infra_error"):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                calls, run = self.fake_backend(root, [(outcome, False)])
                with patch("worldline.runtime.batch.evaluate_native", side_effect=run):
                    evaluate_many_native([TASK], runs_dir=root, agent_factory=None, agent_name="test")
                self.assertEqual(len(calls), 1)

    def test_resume_skips_completed_and_recovers_interrupted_commit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            calls, run = self.fake_backend(root, [("success", False)])
            with patch("worldline.runtime.batch.evaluate_native", side_effect=run):
                batch = evaluate_many_native(
                    [TASK], runs_dir=root, batch_id="resume", agent_factory=None, agent_name="test"
                )
                # Simulate process death after the durable verdict but before queue commit.
                with sqlite3.connect(batch["state"]) as db:
                    db.execute("UPDATE jobs SET status='running'")
                resumed = evaluate_many_native(
                    [TASK], runs_dir=root, batch_id="resume", resume=True, agent_factory=None, agent_name="test"
                )
            self.assertEqual(len(calls), 1)
            self.assertEqual(resumed["statuses"], {"completed": 1})
            with self.assertRaisesRegex(ValueError, "differs"):
                evaluate_many_native(
                    [TASK], runs_dir=root, batch_id="resume", resume=True, agent_factory=None, agent_name="different"
                )

    def test_resource_capacity_rejects_unschedulable_jobs(self):
        pool = ResourcePool(2, ["0"])
        device = pool.acquire(1, True)
        self.assertEqual(device, "0")
        pool.release(1, device)
        with self.assertRaises(ValueError):
            pool.acquire(3, False)

    def test_bad_credentials_block_remaining_jobs_without_repeated_calls(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            calls, backend = self.fake_backend(root, [("infra_error", False)])

            def failed(path, **kwargs):
                record, directory = backend(path, **kwargs)
                result_path = directory / "result.normalized.json"
                result = json.loads(result_path.read_text())
                result["error_code"] = "model.http.401"
                write_json(result_path, result)
                return record, directory

            with patch("worldline.runtime.batch.evaluate_native", side_effect=failed):
                manifest = evaluate_many_native(
                    [TASK, TASK], runs_dir=root, max_workers=1, agent_factory=None, agent_name="openai-compatible"
                )
            self.assertEqual(len(calls), 1)
            self.assertEqual(manifest["statuses"], {"infra_error": 1, "blocked": 1})
            self.assertEqual(next(iter(build_report([root])["batches"].values()))["unscored_jobs"], 2)
