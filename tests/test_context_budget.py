from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from unittest.mock import patch

from worldline.agents.context import bounded_context
from worldline.agents.endpoint import EndpointLimiter
from worldline.agents.tool_agent import OpenAICompatibleAgent
from worldline.errors import BudgetExceeded, ModelAPIError


class ContextBudgetTest(unittest.TestCase):
    def test_context_removes_whole_turns_and_preserves_archive(self):
        initial = [{"role": "system", "content": "instructions"}, {"role": "user", "content": "goal"}]
        old = [
            {"role": "assistant", "content": "x" * 1000, "tool_calls": [{"id": "old"}]},
            {"role": "tool", "tool_call_id": "old", "content": "old reply"},
        ]
        recent = [
            {"role": "assistant", "content": "new", "tool_calls": [{"id": "new"}]},
            {"role": "tool", "tool_call_id": "new", "content": "new reply"},
        ]
        messages = initial + old + recent
        compact, removed = bounded_context(messages, 500)
        self.assertEqual(compact, initial + recent)
        self.assertEqual(removed, 2)
        self.assertEqual(len(messages), 6)
        with self.assertRaises(BudgetExceeded):
            bounded_context(messages, 10)

    def agent(self, **options):
        return OpenAICompatibleAgent(
            model="model",
            base_url="http://localhost:9999/v1",
            spec={"action_schema": {}},
            initial_observation={},
            emit=lambda *_: None,
            messages=[{"role": "user", "content": "goal"}],
            usage={},
            max_turns=5,
            deadline=time.monotonic() + 3,
            **options,
        )

    def test_sampling_parameters_and_token_stop_are_effective(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"WORLDLINE_STATE_DIR": temp}):
            agent = self.agent(max_total_tokens=5, max_output_tokens=4, sampling_seed=7)
            result = {"usage": {"total_tokens": 5}, "choices": [{"message": {"role": "assistant", "content": "done"}}]}
            with patch(
                "worldline.agents.tool_agent.http_completion",
                return_value=result,
            ) as request:
                agent._completion()
                body = json.loads(request.call_args.args[0].data)
                self.assertEqual(body["max_tokens"], 4)
                self.assertEqual(body["seed"], 7)
            with self.assertRaises(BudgetExceeded):
                agent._completion()

    def test_token_budget_requires_usage_and_response_size_is_bounded(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"WORLDLINE_STATE_DIR": temp}):
            agent = self.agent(max_total_tokens=10)
            with patch("worldline.agents.tool_agent.http_completion", return_value={"choices": []}):
                with self.assertRaisesRegex(ModelAPIError, "report token usage"):
                    agent._completion()
            agent = self.agent(max_response_bytes=4)
            with patch(
                "worldline.agents.tool_agent.http_completion",
                side_effect=ModelAPIError("model response exceeds byte budget", status=422),
            ):
                with self.assertRaisesRegex(ModelAPIError, "byte budget"):
                    agent._completion()

    def test_endpoint_admission_deadline_and_policy_mismatch(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"WORLDLINE_STATE_DIR": temp}):
            first = EndpointLimiter("endpoint", concurrency=1)
            second = EndpointLimiter("endpoint", concurrency=1)
            with first.request(time.monotonic() + 2):
                with self.assertRaises(BudgetExceeded):
                    with second.request(time.monotonic() + 0.05):
                        self.fail("two endpoint slots were admitted")
            with self.assertRaises(ValueError):
                EndpointLimiter("endpoint", concurrency=2)
