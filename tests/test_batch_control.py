from __future__ import annotations

import tempfile
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from worldline.runtime.batch import evaluate_many_native
from worldline.runtime.queue import batch_status, request_stop
from worldline.runtime.resources import ResourcePool
import test_infra_reliability as reliability


class BatchControlTest(unittest.TestCase):
    def test_drain_preserves_pending_job_and_resume_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            calls, backend = reliability.BatchReliabilityTest().fake_backend(
                root, [("success", False), ("success", False)]
            )
            observed = []

            def run(path, **kwargs):
                result = backend(path, **kwargs)
                if len(calls) == 1:
                    observed.append(request_stop("drain", root))
                return result

            with patch("worldline.runtime.batch.evaluate_native", side_effect=run):
                batch = evaluate_many_native(
                    [reliability.TASK, reliability.TASK],
                    runs_dir=root,
                    batch_id="drain",
                    agent_factory=None,
                    agent_name="test",
                )
                status = batch_status("drain", root)
                self.assertEqual(status["statuses"], {"completed": 1, "pending": 1})
                self.assertFalse(status["owner_active"])
                self.assertTrue(observed[0]["owner_active"])
                self.assertEqual(batch["runs"][1]["attempts"], [])
                resumed = evaluate_many_native(
                    [reliability.TASK, reliability.TASK],
                    runs_dir=root,
                    batch_id="drain",
                    resume=True,
                    agent_factory=None,
                    agent_name="test",
                )
            self.assertEqual(resumed["statuses"], {"completed": 2})
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[1]["attempt"], 1)

    def test_missing_completed_artifact_does_not_get_a_free_rerun(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            calls, backend = reliability.BatchReliabilityTest().fake_backend(root, [("success", False)])
            with patch("worldline.runtime.batch.evaluate_native", side_effect=backend):
                batch = evaluate_many_native(
                    [reliability.TASK], runs_dir=root, batch_id="lost", agent_factory=None, agent_name="test"
                )
                (Path(batch["runs"][0]["run_dir"]) / "result.normalized.json").unlink()
                with self.assertRaisesRegex(ValueError, "artifact missing"):
                    evaluate_many_native(
                        [reliability.TASK],
                        runs_dir=root,
                        batch_id="lost",
                        resume=True,
                        agent_factory=None,
                        agent_name="test",
                    )
            self.assertEqual(len(calls), 1)

    def test_duplicate_gpu_names_cannot_grant_two_exclusive_leases(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            ResourcePool(2, ("0", "0"))

    def test_completed_result_hash_change_is_detected_on_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            calls, backend = reliability.BatchReliabilityTest().fake_backend(root, [("success", False)])
            with patch("worldline.runtime.batch.evaluate_native", side_effect=backend):
                batch = evaluate_many_native(
                    [reliability.TASK], runs_dir=root, batch_id="changed", agent_factory=None, agent_name="test"
                )
                result_path = Path(batch["runs"][0]["run_dir"]) / "result.normalized.json"
                result = json.loads(result_path.read_text())
                result["reward"] = 0.5
                result_path.write_text(json.dumps(result))
                with self.assertRaisesRegex(ValueError, "artifact missing or invalid"):
                    evaluate_many_native(
                        [reliability.TASK],
                        runs_dir=root,
                        batch_id="changed",
                        resume=True,
                        agent_factory=None,
                        agent_name="test",
                    )
            self.assertEqual(len(calls), 1)
