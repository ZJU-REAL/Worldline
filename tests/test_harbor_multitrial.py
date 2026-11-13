from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from worldline.datasets import discover_results, export_dataset
from worldline.eval import _ingest_completed_trials
from worldline.report import build_report
from worldline.trace import TraceWriter


class HarborMultiTrialTest(unittest.TestCase):
    def test_every_trial_is_discovered_reported_and_exported(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            backend, run = root / "harbor", root / "run"
            run.mkdir()
            for name, success in (("one", 1), ("two", 0)):
                trial = backend / name
                (trial / "verifier").mkdir(parents=True)
                (trial / "agent").mkdir()
                (trial / "result.json").write_text(json.dumps({"id": name, "task_name": "task"}))
                (trial / "verifier/reward.json").write_text(json.dumps({"success": success}))
                (trial / "agent/trajectory.json").write_text(
                    json.dumps(
                        {
                            "steps": [
                                {"source": "user", "message": "do the task"},
                                {"source": "agent", "message": "done"},
                            ]
                        }
                    )
                )
            trace = TraceWriter(run / "trace.jsonl")
            self.assertEqual(_ingest_completed_trials(backend, run, trace, "test"), 2)
            trace.close()
            self.assertEqual(len(discover_results([run])), 2)
            self.assertEqual(build_report([run])["summary"]["trials"], 2)
            self.assertEqual(export_dataset([run], root / "rl.jsonl", format="rl")["records"], 0)
            self.assertEqual(
                export_dataset([run], root / "reviewed.jsonl", format="rl", allow_unspecified_split=True)["records"], 2
            )
