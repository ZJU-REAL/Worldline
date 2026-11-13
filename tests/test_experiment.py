from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from worldline.experiment import run_experiment


class ExperimentTest(unittest.TestCase):
    def plan(self, root):
        task = Path(__file__).resolve().parents[1] / "examples/rua/reach-train"
        path = root / "plan.toml"
        path.write_text(
            'schema_version="worldline.experiment/0.1"\noutput_dir="output"\n[[phase]]\nid="run"\ntasks=['
            + json.dumps(str(task))
            + ']\n[phase.model]\nmodel="test"\nbase_url="http://localhost:9999/v1"\n'
        )
        return path

    def test_dry_run_state_is_separate_from_execution_and_repeatable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = self.plan(root)
            with patch("worldline.runtime.session.ProcessEnvironment") as environment:
                self.assertTrue(run_experiment(plan, dry_run=True)["passed"])
                self.assertTrue(run_experiment(plan, dry_run=True)["passed"])
                environment.assert_not_called()
            with patch(
                "worldline.experiment.evaluate_many_native", return_value={"statuses": {"completed": 1}}
            ) as evaluation:
                result = run_experiment(plan)
                self.assertTrue(result["passed"])
                self.assertFalse(evaluation.call_args.kwargs["resume"])
                self.assertIn("executions", str(evaluation.call_args.kwargs["runs_dir"]))
            with self.assertRaisesRegex(ValueError, "resume"):
                run_experiment(plan)

    def test_failure_stage_is_persisted_without_overwriting_another_invocation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = self.plan(root)
            with patch("worldline.experiment.evaluate_many_native", side_effect=RuntimeError("unavailable")):
                with self.assertRaises(RuntimeError):
                    run_experiment(plan)
            manifest = json.loads((root / "output/experiment.json").read_text())
            self.assertFalse(manifest["passed"])
            self.assertEqual(manifest["active_phase"], "run")
            self.assertEqual(manifest["stage"], "evaluation")
            self.assertEqual(manifest["status"], "failed")
