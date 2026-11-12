from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from worldline.datasets import export_dataset
from worldline.domains.cua.reference import FormReferenceAgent
from worldline.runtime.session import evaluate_native
from worldline.learning.handoff import export_handoff
from worldline.adapters.browsergym import BrowserGymEnvironment
from worldline.errors import InvalidAction

TASK = Path(__file__).resolve().parents[1] / "examples/cua/form-train"
try:
    AVAILABLE = importlib.util.find_spec("browsergym.core") is not None
except ModuleNotFoundError:
    AVAILABLE = False


@unittest.skipUnless(AVAILABLE, "install CUA extra and Playwright Chromium")
class BrowserGymIntegrationTest(unittest.TestCase):
    def test_actual_browser_form_submission_trace_and_training_export(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            record, directory = evaluate_native(
                TASK, agent_factory=FormReferenceAgent, agent_name="cua-reference", runs_dir=root
            )
            result = json.loads((directory / "result.normalized.json").read_text())
            self.assertEqual(record.status, "completed", result.get("error"))
            self.assertEqual(result["outcome"], "success", result)
            self.assertEqual(result["metrics"]["browser_steps"], 3)
            self.assertTrue(list((directory / "media").glob("*.png")))
            output = root / "sft.jsonl"
            self.assertEqual(export_dataset([directory], output, format="sft")["records"], 1)
            package = export_handoff(output, root / "llama", framework="llama-factory", template="qwen2_vl")
            self.assertEqual(package["records"]["train"], 1)

    def test_noop_is_not_a_success(self):
        from functools import partial

        with tempfile.TemporaryDirectory() as temp:
            _, directory = evaluate_native(
                TASK, agent_factory=partial(FormReferenceAgent, noop=True), agent_name="noop", runs_dir=Path(temp)
            )
            result = json.loads((directory / "result.normalized.json").read_text())
            self.assertEqual(result["outcome"], "failure", result)


class BrowserActionContractTest(unittest.TestCase):
    def test_navigation_cannot_execute_javascript_or_read_worker_files(self):
        environment = object.__new__(BrowserGymEnvironment)
        environment.terminated = environment.truncated = False
        environment.navigation = True
        for url in ("javascript:alert(1)", "file:///private/verifier.json"):
            with self.assertRaises(InvalidAction):
                environment.step({"op": "goto", "url": url})
