from __future__ import annotations

import base64
import importlib.util
import io
import json
import tempfile
import threading
import unittest
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from worldline.agents.tool_agent import OpenAICompatibleAgent
from worldline.datasets import export_dataset
from worldline.learning.handoff import export_handoff
from worldline.runtime.session import evaluate_native

try:
    AVAILABLE = importlib.util.find_spec("browsergym.core") is not None
except ModuleNotFoundError:
    AVAILABLE = False
TASK = Path(__file__).resolve().parents[1] / "examples/cua/form-train"


@unittest.skipUnless(AVAILABLE, "install CUA extra and Chromium")
class CuaModelPipelineTest(unittest.TestCase):
    def test_api_tool_agent_browser_and_visual_training_data_share_the_pipeline(self):
        requests, actions = [], []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                index = len(requests)
                requests.append(body)
                if index == 0:
                    fields = json.loads(body["messages"][1]["content"].split("Fields: ", 1)[1].strip())
                    state = json.loads(body["messages"][2]["content"][0]["text"])
                    for name, text in fields.items():
                        bid = next(
                            item["bid"]
                            for item in state["elements"]
                            if item["role"] == "textbox" and item["name"] == name
                        )
                        actions.append({"op": "fill", "bid": bid, "text": text})
                    bid = next(
                        item["bid"]
                        for item in state["elements"]
                        if item["role"] == "button" and item["name"] == "Submit project"
                    )
                    actions.append({"op": "click", "bid": bid})
                message = {"role": "assistant", "content": "done"}
                if index < len(actions):
                    message = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": str(index),
                                "type": "function",
                                "function": {"name": "act", "arguments": json.dumps({"action": actions[index]})},
                            }
                        ],
                    }
                self.send_response(200)
                self.end_headers()
                self.wfile.write(
                    json.dumps(
                        {"choices": [{"message": message}], "usage": {"prompt_tokens": 3, "completion_tokens": 2}}
                    ).encode()
                )

            def log_message(self, *_):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            agent = partial(
                OpenAICompatibleAgent,
                model="fixture-policy",
                base_url=f"http://127.0.0.1:{server.server_port}/v1",
                max_context_images=2,
            )
            record, run = evaluate_native(TASK, agent_factory=agent, agent_name="openai-compatible", runs_dir=root)
            result = json.loads((run / "result.normalized.json").read_text())
            self.assertEqual(record.status, "completed", result.get("error"))
            self.assertEqual(result["outcome"], "success")
            self.assertEqual(result["model"], "fixture-policy")
            self.assertEqual(record.metadata["agent_runtime_config"]["settings"]["max_context_images"], 2)
            self.assertEqual(len(requests), 4)
            from PIL import Image

            for request in requests:
                frames = [
                    part
                    for msg in request["messages"]
                    if isinstance(msg.get("content"), list)
                    for part in msg["content"]
                    if part["type"] == "image_url"
                ]
                self.assertLessEqual(len(frames), 2)
                for part in frames:
                    encoded = part["image_url"]["url"].split(",", 1)[1]
                    Image.open(io.BytesIO(base64.b64decode(encoded))).verify()
            data = root / "sft.jsonl"
            export_dataset([run], data, format="sft", media_mode="reference")
            manifest = export_handoff(data, root / "lf", framework="llama-factory")
            self.assertEqual(manifest["records"]["train"], 1)
            row = json.loads((root / "lf/data/train.jsonl").read_text())
            self.assertEqual(len(row["images"]), 4)
