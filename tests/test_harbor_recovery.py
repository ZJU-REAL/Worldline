from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from worldline.batch import evaluate_many_harbor
from worldline.runtime.queue import request_stop, batch_status
from worldline.run import RunRecord, write_record, write_json
from test_batch import task


class HarborRecoveryTest(unittest.TestCase):
    def backend(self, root, calls, stop=False):
        def run(path, **options):
            ordinal = len(calls)
            calls.append(path)
            directory = root / f"run-{ordinal}"
            directory.mkdir()
            record = RunRecord(str(ordinal), f"example/{path.name}", "v1", "harbor", "test", None, (), "created")
            write_record(directory, record)
            write_json(
                directory / "task.json",
                {"id": record.task_id, "version": "v1", "seed": 0, "split": "train", "metadata": {}},
            )
            options["on_created"](record, directory)
            record = replace(record, status="completed", finished_at_ns=1)
            write_record(directory, record)
            if stop:
                request_stop("harbor", root)
            return record, directory

        return run

    def test_completed_skip_and_drain_use_the_same_durable_queue(self):
        with tempfile.TemporaryDirectory() as temp:
            root, calls = Path(temp), []
            tasks = [task(root, "one"), task(root, "two")]
            with patch("worldline.batch.evaluate_harbor", side_effect=self.backend(root, calls, stop=True)):
                first = evaluate_many_harbor(tasks, agent="test", runs_dir=root, batch_id="harbor")
            self.assertEqual(first["statuses"], {"completed": 1, "pending": 1})
            with patch("worldline.batch.evaluate_harbor", side_effect=self.backend(root, calls)):
                final = evaluate_many_harbor(tasks, agent="test", runs_dir=root, batch_id="harbor", resume=True)
            self.assertEqual(len(calls), 2)
            self.assertEqual(final["statuses"], {"completed": 2})
            self.assertFalse(batch_status("harbor", root)["owner_active"])

    def test_backend_completed_before_queue_commit_is_recovered_without_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            root, calls = Path(temp), []
            tasks = [task(root, "one")]
            with patch("worldline.batch.evaluate_harbor", side_effect=self.backend(root, calls)):
                first = evaluate_many_harbor(tasks, agent="test", runs_dir=root, batch_id="recover")
                with sqlite3.connect(first["state"]) as db:
                    db.execute("UPDATE jobs SET status='running'")
                resumed = evaluate_many_harbor(tasks, agent="test", runs_dir=root, batch_id="recover", resume=True)
            self.assertEqual(len(calls), 1)
            self.assertEqual(resumed["statuses"], {"completed": 1})

    def test_ambiguous_started_trial_is_preserved_and_later_completion_can_be_ingested(self):
        with tempfile.TemporaryDirectory() as temp:
            root, calls = Path(temp), []
            tasks = [task(root, "one")]
            with patch("worldline.batch.evaluate_harbor", side_effect=self.backend(root, calls)):
                first = evaluate_many_harbor(tasks, agent="test", runs_dir=root, batch_id="orphan")
                directory = Path(first["runs"][0]["run_dir"])
                saved = json.loads((directory / "run.json").read_text())
                saved["status"] = "running"
                (directory / "run.json").write_text(json.dumps(saved))
                with sqlite3.connect(first["state"]) as db:
                    db.execute("UPDATE jobs SET status='running'")
                orphan = evaluate_many_harbor(tasks, agent="test", runs_dir=root, batch_id="orphan", resume=True)
                self.assertEqual(orphan["statuses"], {"orphaned": 1})
                write_json(directory / "harbor/job/result.json", {"finished_at": "2026-11-13T00:00:00Z"})
                finished = evaluate_many_harbor(tasks, agent="test", runs_dir=root, batch_id="orphan", resume=True)
            self.assertEqual(len(calls), 1)
            self.assertEqual(finished["statuses"], {"completed": 1})

    def test_resume_rejects_changed_configuration(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tasks = [task(root, "one")]
            evaluate_many_harbor(tasks, agent="test", runs_dir=root, batch_id="config", dry_run=True)
            with self.assertRaisesRegex(ValueError, "differs"):
                evaluate_many_harbor(
                    tasks, agent="another", runs_dir=root, batch_id="config", dry_run=True, resume=True
                )

            (tasks[0] / "instruction.md").write_text("changed task")
            with self.assertRaisesRegex(ValueError, "differs"):
                evaluate_many_harbor(tasks, agent="test", runs_dir=root, batch_id="config", dry_run=True, resume=True)
