from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from worldline.contracts import Task
from worldline.domains.rua.mujoco_reach import MujocoReach
from worldline.runtime.interactive import InteractiveSession
from worldline.errors import BudgetExceeded
from test_interactive_training import tool

TASK = Path(__file__).resolve().parents[1] / "examples/rua/reach-train"


@unittest.skipUnless(importlib.util.find_spec("mujoco"), "install rua extra")
class CheckpointTest(unittest.TestCase):
    def test_restores_physics_and_identical_next_action_without_rewinding_compute(self):
        task = Task("task", "task", "rua.mujoco-reach", "v1", seed=7)
        environment = MujocoReach(task, {})
        try:
            environment.step({"joint_targets": [0.5, -0.5]})
            before = environment.observe()
            checkpoint = json.loads(json.dumps(environment.checkpoint()))
            first = environment.step({"joint_targets": [0.4, -0.3]})
            first_state = environment.checkpoint()["state"]
            self.assertEqual(environment.restore(checkpoint), before)
            second = environment.step({"joint_targets": [0.4, -0.3]})
            self.assertEqual(second, first)
            self.assertEqual(environment.checkpoint()["state"], first_state)
            self.assertEqual(environment.verify().metrics["sim_steps_total"], 150)
        finally:
            environment.close()

    def test_checkpoint_restores_the_rendered_observation_in_a_fresh_worker(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "visual"
            shutil.copytree(TASK.parent / "reach-visual-cpu", path)
            manifest = path / "worldline.toml"
            manifest.write_text(
                manifest.read_text().replace("reset_budget = 0", "reset_budget = 0\nrestore_budget = 1")
            )
            with InteractiveSession(path, runs_dir=root / "runs") as session:
                checkpoint = session.checkpoint()
                before = session.environment.request("observe")
                session.advance(tool(1, "act", {"action": {"joint_targets": [0.6, 0.3]}}))
                restored = session.restore(checkpoint)
                self.assertEqual(restored, before)
                self.assertTrue(restored["images"])

    def test_invalid_state_is_rejected_before_mutating_environment(self):
        environment = MujocoReach(Task("task", "task", "rua.mujoco-reach", "v1"), {})
        try:
            checkpoint = environment.checkpoint()
            corrupt = copy.deepcopy(checkpoint)
            corrupt["state"][0] = float("nan")
            before = environment.observe()
            with self.assertRaises(ValueError):
                environment.restore(corrupt)
            self.assertEqual(environment.observe(), before)
        finally:
            environment.close()

    def test_controller_restore_uses_declared_budget_and_does_not_reset_action_count(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "task"
            shutil.copytree(TASK, path)
            manifest = path / "worldline.toml"
            manifest.write_text(
                manifest.read_text().replace("reset_budget = 0", "reset_budget = 0\nrestore_budget = 1")
            )
            with InteractiveSession(path, runs_dir=root / "runs") as session:
                session.advance(tool(1, "act", {"action": {"joint_targets": [0.4, 0.3]}}))
                checkpoint = session.checkpoint()
                before = session.environment.request("observe")
                session.advance(tool(2, "act", {"action": {"joint_targets": [-0.4, -0.3]}}))
                restored = session.restore(checkpoint)
                self.assertEqual(restored, before)
                self.assertEqual(session.view.actions, 2)
                self.assertEqual(session.restores, 1)
                with self.assertRaises(BudgetExceeded):
                    session.restore(checkpoint)
                result = session.finish()
                self.assertEqual(result.metrics["n_restores"], 1)
                self.assertFalse(session.view.capabilities.snapshot)
                self.assertTrue(list((session.run_dir / "checkpoints").glob("*.json")))
