from __future__ import annotations

import importlib.util
import json
import math
import tempfile
import shutil
import time
import unittest
from pathlib import Path

from worldline.learning.rollout import NativeTrainingSession
from worldline.runtime.interactive import InteractiveSession
from worldline.errors import BudgetExceeded
from worldline.runtime.resources import ResourcePool

TASK = Path(__file__).resolve().parents[1] / "examples/rua/reach-train"


def tool(index, name, arguments):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": str(index), "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
        ],
    }


@unittest.skipUnless(importlib.util.find_spec("mujoco"), "install rua extra")
class InteractiveTrainingTest(unittest.TestCase):
    def test_idle_controller_deadline_reaps_worker_and_releases_resources(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            task_dir = root / "task"
            shutil.copytree(TASK, task_dir)
            manifest = task_dir / "worldline.toml"
            manifest.write_text(manifest.read_text().replace("timeout_sec = 120", "timeout_sec = 2"))
            pool = ResourcePool(1, ())
            session = NativeTrainingSession(task_dir, runs_dir=root / "runs", resources=pool)
            session.start()
            limit = time.monotonic() + 5
            while session.result is None and time.monotonic() < limit:
                time.sleep(0.02)
            try:
                self.assertIsNotNone(session.result)
                self.assertEqual(session.result.outcome, "failure")
                self.assertEqual(pool.available, 1)
                self.assertIsNotNone(session.environment.process.poll())
                with self.assertRaises(BudgetExceeded):
                    session.advance({"role": "assistant", "content": "late completion"})
            finally:
                session.abort()

    def test_training_session_releases_shared_resources_on_success_and_abort(self):
        pool = ResourcePool(1, ())
        with tempfile.TemporaryDirectory() as temp:
            with NativeTrainingSession(TASK, runs_dir=Path(temp), resources=pool) as session:
                self.assertEqual(pool.available, 0)
                session.abort(RuntimeError("provider failed"))
            self.assertEqual(pool.available, 1)
            with NativeTrainingSession(TASK, runs_dir=Path(temp), resources=pool) as session:
                self.assertEqual(pool.available, 0)
                session.finish()
            self.assertEqual(pool.available, 1)

    def test_new_policy_actions_get_a_live_reward_and_durable_trace(self):
        with tempfile.TemporaryDirectory() as temp:
            session = NativeTrainingSession(TASK, runs_dir=Path(temp))
            state = session.start()
            x, y = state["observation"]["target_position"]
            first, second = state["spec"]["link_lengths"]
            elbow = math.acos((x * x + y * y - first * first - second * second) / (2 * first * second))
            shoulder = math.atan2(y, x) - math.atan2(second * math.sin(elbow), first + second * math.cos(elbow))
            try:
                for index in range(20):
                    responses = session.advance(tool(index, "act", {"action": {"joint_targets": [shoulder, elbow]}}))
                    observation = json.loads(responses[0]["content"])
                    if (
                        math.dist(observation["tip_position"], [x, y]) < 0.015
                        and math.hypot(*observation["joint_velocities"]) < 0.1
                    ):
                        break
                session.advance(tool("finish", "finish", {}))
                outcome = session.training_outcome()
                self.assertTrue(outcome.valid)
                self.assertEqual(outcome.reward, 1)
                self.assertEqual(session.training_outcome(), outcome)
                self.assertEqual(
                    json.loads((session.run_dir / "result.normalized.json").read_text())["outcome"], "success"
                )
                self.assertIsNotNone(session.environment.process.poll())
            finally:
                session.abort()

    def test_noop_is_trainable_failure_and_abandonment_is_unscorable(self):
        with tempfile.TemporaryDirectory() as temp:
            with NativeTrainingSession(TASK, runs_dir=Path(temp)) as session:
                session.advance({"role": "assistant", "content": "I am done."})
                outcome = session.training_outcome()
                self.assertTrue(outcome.valid)
                self.assertEqual(outcome.reward, 0)
            with NativeTrainingSession(TASK, runs_dir=Path(temp)) as aborted:
                aborted.abort(RuntimeError("model server unavailable"))
                outcome = aborted.training_outcome()
                self.assertFalse(outcome.valid)
                self.assertIsNone(outcome.reward)

    def test_training_rejects_test_tasks_and_verifier_tool_is_not_public(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "train task"):
                NativeTrainingSession(TASK.parent / "reach-test", runs_dir=Path(temp))
            with NativeTrainingSession(TASK, runs_dir=Path(temp)) as session:
                responses = session.advance(tool("forged", "verify", {"reward": 1}))
                self.assertIn("unknown tool", responses[0]["content"])
                self.assertEqual(session.training_outcome().reward, 0)

    def test_budget_exhaustion_still_grades_actual_final_state(self):
        with tempfile.TemporaryDirectory() as temp:
            with InteractiveSession(TASK, runs_dir=Path(temp)) as session:
                session.view.actions = session.definition.max_actions
                with self.assertRaises(BudgetExceeded):
                    session.advance(tool("overflow", "act", {"action": {"joint_targets": [0, 0]}}))
                self.assertEqual(session.finish().outcome, "failure")
