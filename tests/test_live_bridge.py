from __future__ import annotations

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from worldline.errors import BudgetExceeded
from worldline.learning.bridge import EpisodeRegistry
from worldline.learning.rollout import NativeTrainingSession
from worldline.runtime.resources import ResourcePool

TASK = Path(__file__).resolve().parents[1] / "examples/rua/reach-train"


def action(targets, ident="call"):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": ident,
                "type": "function",
                "function": {"name": "act", "arguments": json.dumps({"action": {"joint_targets": targets}})},
            }
        ],
    }


@unittest.skipUnless(importlib.util.find_spec("mujoco"), "install rua extra")
class LiveBridgeTest(unittest.TestCase):
    def test_per_tool_release_preserves_episode_and_duplicate_request_does_not_repeat_action(self):
        with tempfile.TemporaryDirectory() as temp:
            registry = EpisodeRegistry({"reach": TASK}, runs_dir=Path(temp), resources=ResourcePool(1, ()))
            try:
                start = registry.begin("reach")
                episode = start["episode_id"]
                result = registry.advance(episode, action([0.1, 0.2]), request_id="one")
                registry.release_tool(episode)
                self.assertEqual(registry.advance(episode, action([0.1, 0.2]), request_id="one"), result)
                self.assertEqual(registry.sessions[episode].view.actions, 1)
                with self.assertRaisesRegex(ValueError, "different content"):
                    registry.advance(episode, action([0.3, 0.4]), request_id="one")
                registry.advance(episode, action([0.2, 0.1], "two"), request_id="two")
                self.assertEqual(registry.sessions[episode].view.actions, 2)
                outcome = registry.finish(episode)
                self.assertTrue(outcome["valid"])
                self.assertEqual(outcome["reward"], 0)
            finally:
                registry.close()

    def test_fork_has_independent_physics_lineage_and_original_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            task = root / "task"
            shutil.copytree(TASK, task)
            manifest = task / "worldline.toml"
            manifest.write_text(manifest.read_text().replace("[evaluation]", "[evaluation]\nfork_budget = 1"))
            parent = NativeTrainingSession(task, runs_dir=root / "runs", resources=ResourcePool(2, ()))
            child = None
            try:
                parent.start()
                parent.advance(action([0.1, 0.2]))
                checkpoint = parent.checkpoint()
                original = parent.view.observe()
                child = parent.fork(checkpoint)
                self.assertNotEqual(parent.environment.process.pid, child.environment.process.pid)
                self.assertEqual(child.environment.deadline, parent.environment.deadline)
                self.assertEqual(child.view.observe(), original)
                child.advance(action([0.8, 0.6], "child"))
                self.assertEqual(parent.view.observe(), original)
                self.assertNotEqual(child.view.observe(), original)
                self.assertEqual(child.view.actions, 2)
                with self.assertRaises(BudgetExceeded):
                    parent.fork(checkpoint)
                with self.assertRaises(BudgetExceeded):
                    child.fork()
                child.training_outcome()
                record = json.loads((child.run_dir / "run.json").read_text())
                self.assertEqual(record["metadata"]["lineage"]["parent_run_id"], parent.run_id)
                result = json.loads((child.run_dir / "result.normalized.json").read_text())
                self.assertGreaterEqual(result["metrics"]["sim_steps_total"], 100)
                self.assertEqual(result["metrics"]["sim_steps_executed"], 50)
                self.assertEqual(result["metrics"]["n_actions_executed"], 1)
            finally:
                if child and child.result is None:
                    child.abort()
                parent.abort()
