from __future__ import annotations

import importlib
import importlib.util
import asyncio
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from worldline.learning.bridge import EpisodeRegistry
from worldline.learning.service import EpisodeClient, server_for
from worldline.runtime.resources import ResourcePool


@unittest.skipUnless(importlib.util.find_spec("mujoco"), "install rua extra")
class TrainingServiceTest(unittest.TestCase):
    def test_verl_style_per_call_handles_share_one_real_environment_episode(self):
        task = Path(__file__).resolve().parents[1] / "examples/rua/reach-train"
        with tempfile.TemporaryDirectory() as temp:
            registry = EpisodeRegistry(
                {"reach": task}, runs_dir=Path(temp), resources=ResourcePool(1, ()), max_sessions=1
            )
            server = server_for(registry, "127.0.0.1", 0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            endpoint = f"http://127.0.0.1:{server.server_port}"
            client = EpisodeClient(endpoint)
            try:
                initial = client.call("begin", task_key="reach")
                episode = initial["episode_id"]
                with self.assertRaises(RuntimeError):
                    registry.begin("reach")
                # Exercise adapter lifecycle without claiming a verl trainer run.
                base = types.ModuleType("verl.tools.base_tool")
                schemas = types.ModuleType("verl.tools.schemas")

                class BaseTool:
                    def __init__(self, *_):
                        pass

                class Response:
                    def __init__(self, text=None, image=None):
                        self.text, self.image = text, image

                base.BaseTool, schemas.ToolResponse = BaseTool, Response
                with patch.dict("sys.modules", {"verl.tools.base_tool": base, "verl.tools.schemas": schemas}):
                    module = importlib.import_module("worldline.learning.verl_tools")
                    tool = module.WorldlineEnvironmentTool({"endpoint": endpoint}, None)

                    async def run():
                        for targets in ([0.1, 0.2], [0.2, 0.1]):
                            instance, _ = await tool.create(create_kwargs={"episode_id": episode})
                            response, reward, _ = await tool.execute(
                                instance, {"op": "act", "action": {"joint_targets": targets}}
                            )
                            self.assertNotIn('"reward"', response.text)
                            self.assertEqual(reward, 0)
                            await tool.release(instance)
                        self.assertEqual(tool.instances, {})

                    asyncio.run(run())
                self.assertEqual(registry.sessions[episode].view.actions, 2)
                outcome = client.call("finish", episode_id=episode)
                self.assertTrue(outcome["valid"])
                self.assertEqual(client.call("finish", episode_id=episode), outcome)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)
                registry.close()
