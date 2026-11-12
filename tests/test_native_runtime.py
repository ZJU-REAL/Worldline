from __future__ import annotations

import importlib.util
import json
import os
import signal
import shutil
import tempfile
import time
import unittest
from dataclasses import replace
from functools import partial
from pathlib import Path
from unittest.mock import patch

from worldline.contracts import Sandbox, Verifier
from worldline.datasets import export_dataset
from worldline.domains.rua.reference import ReachReferenceAgent
from worldline.errors import BudgetExceeded, InvalidAction
from worldline.plugins import EnvironmentPlugin, environment_plugins
from worldline.runtime.batch import evaluate_many_native
from worldline.runtime.process import AgentView, ProcessEnvironment
from worldline.runtime.session import evaluate_native
from worldline.tasks import load_native_task

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/rua/reach-train"
MUJOCO = importlib.util.find_spec("mujoco") is not None


class NativeTaskTest(unittest.TestCase):
    def test_version_changes_with_task_or_environment(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "task"
            shutil.copytree(EXAMPLE, path)
            first = load_native_task(path)
            (path / "instruction.md").write_text("different instruction")
            self.assertNotEqual(first.task.version, load_native_task(path).task.version)
            plugin = EnvironmentPlugin("rua.mujoco-reach", "rua", "new-version", lambda *_: None)
            with patch("worldline.tasks.get_environment_plugin", return_value=plugin):
                self.assertNotEqual(load_native_task(path).task.version, first.task.version)

    def test_plugin_conflicts_and_other_domains(self):
        class Entry:
            name = "cua.example"

            def load(self):
                return lambda: EnvironmentPlugin(self.name, "cua", "1", lambda *_: None)

        with patch("worldline.plugins.entry_points", return_value=[Entry()]):
            self.assertEqual(environment_plugins()["cua.example"].domain, "cua")
        entry = Entry()
        entry.name = "rua.socket"
        with patch("worldline.plugins.entry_points", return_value=[entry]):
            with self.assertRaisesRegex(ValueError, "duplicate"):
                environment_plugins()

    def test_dry_run_never_starts_an_environment(self):
        with tempfile.TemporaryDirectory() as temp, patch("worldline.runtime.session.ProcessEnvironment") as worker:
            record, path = evaluate_native(
                EXAMPLE, agent_factory=lambda **_: None, agent_name="test", runs_dir=Path(temp), dry_run=True
            )
            self.assertEqual(record.status, "planned")
            self.assertTrue((path / "task.json").is_file())
            worker.assert_not_called()


@unittest.skipUnless(MUJOCO, "install Worldline's rua extra to run real simulator tests")
class MujocoRuntimeTest(unittest.TestCase):
    def test_provider_failure_after_verified_success_does_not_void_success(self):
        from worldline.errors import ModelAPIError

        class LateProviderFailure(ReachReferenceAgent):
            def run(self, task, sandbox):
                super().run(task, sandbox)
                self.messages.pop()  # The last tool observation is still a valid recorded outcome.
                raise ModelAPIError("late provider outage", status=503)

        with tempfile.TemporaryDirectory() as temp:
            _, directory = evaluate_native(
                EXAMPLE, agent_factory=LateProviderFailure, agent_name="test", runs_dir=Path(temp)
            )
            result = json.loads((directory / "result.normalized.json").read_text())
            self.assertEqual(result["outcome"], "success")
            self.assertFalse(result["retryable"])

    def test_stuck_worker_is_killed_at_request_deadline(self):
        with tempfile.TemporaryDirectory() as temp:
            worker = ProcessEnvironment(load_native_task(EXAMPLE), Path(temp))
            try:
                worker.start()
                os.kill(worker.process.pid, signal.SIGSTOP)
                worker.deadline = time.monotonic() + 0.05
                with self.assertRaises(BudgetExceeded):
                    worker.request("observe")
                self.assertIsNotNone(worker.process.poll())
            finally:
                worker.close()

    def test_real_rollout_verdict_export_and_worker_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            processes = []
            real = ProcessEnvironment

            def create(*args):
                worker = real(*args)
                processes.append(worker)
                return worker

            with patch("worldline.runtime.session.ProcessEnvironment", side_effect=create):
                record, path = evaluate_native(
                    EXAMPLE, agent_factory=ReachReferenceAgent, agent_name="reference", runs_dir=Path(temp)
                )
            result = json.loads((path / "result.normalized.json").read_text())
            self.assertEqual(record.status, "completed", result.get("error"))
            self.assertEqual(result["outcome"], "success", result)
            self.assertGreater(result["metrics"]["n_actions"], 0)
            self.assertLess(result["metrics"]["distance_m"], 0.025)
            self.assertTrue(all(p.process.poll() is not None for p in processes))
            self.assertTrue(all(p.process.returncode == 0 for p in processes))
            self.assertTrue((path / "verifier/final.json").is_file())
            output = Path(temp) / "sft.jsonl"
            self.assertEqual(export_dataset([path], output, format="sft")["records"], 1)
            row = json.loads(output.read_text())
            self.assertTrue(any(m.get("tool_calls") for m in row["messages"]))
            self.assertTrue(row["tools"])
            self.assertEqual(row["metadata"]["seed"], 7)

    def test_agent_self_report_is_not_a_reward(self):
        class Spoof:
            def run(self, *_):
                return {"success": True, "reward": 1}

        with tempfile.TemporaryDirectory() as temp:
            _, path = evaluate_native(
                EXAMPLE, agent_factory=lambda **_: Spoof(), agent_name="spoof", runs_dir=Path(temp)
            )
            result = json.loads((path / "result.normalized.json").read_text())
            self.assertEqual(result["reward"], 0)
            self.assertEqual(result["outcome"], "failure")

    def test_limits_reset_determinism_and_contract_boundary(self):
        with tempfile.TemporaryDirectory() as temp:
            definition = replace(load_native_task(EXAMPLE), reset_budget=1, max_actions=1)
            worker = ProcessEnvironment(definition, Path(temp))
            try:
                initial = worker.start()
                view = AgentView(worker, lambda *_: None)
                self.assertIsInstance(view, Sandbox)
                self.assertIsInstance(worker, Verifier)
                self.assertFalse(hasattr(view, "verify"))
                self.assertFalse(view.capabilities.snapshot)
                view.act({"joint_targets": [0.1, 0.2]})
                self.assertEqual(view.reset(), initial)
                with self.assertRaises(InvalidAction):
                    view.reset()
                with self.assertRaises(BudgetExceeded):
                    view.act({"joint_targets": [0.1, 0.2]})
                view.close()
                with self.assertRaises(BudgetExceeded):
                    view.observe()
            finally:
                worker.close()

    def test_nonfinite_action_and_worker_failure_are_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            definition = load_native_task(EXAMPLE)
            worker = ProcessEnvironment(definition, Path(temp))
            try:
                worker.start()
                view = AgentView(worker, lambda *_: None)
                with self.assertRaises((InvalidAction, ValueError)):
                    view.act({"joint_targets": [float("nan"), 0]})
                with self.assertRaises(InvalidAction):
                    view.act({"joint_targets": [True, 0]})
            finally:
                worker.close()
            broken = Path(temp) / "broken"
            shutil.copytree(EXAMPLE, broken)
            manifest = broken / "worldline.toml"
            manifest.write_text(manifest.read_text().replace("[0.4, 0.25]", "[4.0, 4.0]"))
            _, path = evaluate_native(
                broken, agent_factory=ReachReferenceAgent, agent_name="reference", runs_dir=Path(temp)
            )
            self.assertEqual(json.loads((path / "result.normalized.json").read_text())["outcome"], "infra_error")

    def test_deadline_stops_agent_but_still_verifies_final_state(self):
        class Late:
            def run(self, task, view):
                view._environment.deadline = time.monotonic() - 1
                view.observe()

        with tempfile.TemporaryDirectory() as temp:
            _, path = evaluate_native(EXAMPLE, agent_factory=lambda **_: Late(), agent_name="late", runs_dir=Path(temp))
            result = json.loads((path / "result.normalized.json").read_text())
            self.assertEqual(result["outcome"], "failure")
            self.assertIn("BudgetExceeded", result["error"])

    def test_batch_failure_does_not_abort_other_tasks(self):
        with tempfile.TemporaryDirectory() as temp:
            result = evaluate_many_native(
                [EXAMPLE, Path(temp) / "missing"],
                agent_factory=partial(ReachReferenceAgent, noop=True),
                agent_name="noop",
                runs_dir=Path(temp),
                max_workers=2,
            )
            self.assertEqual(result["statuses"], {"completed": 1, "error": 1})
