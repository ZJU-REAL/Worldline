"""A real, CPU-capable MuJoCo integration fixture, not a benchmark score."""

from __future__ import annotations

import base64
import hashlib
import io
import math
import random
from pathlib import Path
from typing import Any, Mapping

from worldline.contracts import RunResult, Task
from worldline.errors import InvalidAction


class MujocoReach:
    @staticmethod
    def version() -> str:
        folder = Path(__file__).parent
        content = (folder / "arm.xml").read_bytes() + Path(__file__).read_bytes()
        return "mujoco-reach/0.1:" + hashlib.sha256(content).hexdigest()

    def __init__(self, task: Task, config: Mapping[str, Any]) -> None:
        import mujoco
        import numpy as np

        self.mujoco, self.np = mujoco, np
        self.model = mujoco.MjModel.from_xml_path(str(Path(__file__).with_name("arm.xml")))
        self.data = mujoco.MjData(self.model)
        self.target = [float(v) for v in config.get("target", [0.4, 0.25])]
        if len(self.target) != 2 or not all(math.isfinite(v) for v in self.target):
            raise ValueError("target must contain two finite coordinates")
        if not 0.15 <= math.hypot(*self.target) <= 0.65:
            raise ValueError("target must be within the arm's reachable annulus [0.15, 0.65] m")
        self.threshold = float(config.get("threshold_m", 0.025))
        if not 0 < self.threshold <= 0.05:
            raise ValueError("threshold_m must be in (0, 0.05]")
        self.frame_skip = config.get("frame_skip", 50)
        if type(self.frame_skip) is not int or not 1 <= self.frame_skip <= 500:
            raise ValueError("frame_skip must be an integer in [1, 500]")
        self.renderer = mujoco.Renderer(self.model, height=256, width=256) if config.get("render", False) else None
        self.steps = 0
        self.reset(task.seed)

    def spec(self) -> Mapping[str, Any]:
        return {
            "protocol": "worldline.rua/0.1",
            "robot": "planar-two-link-arm",
            "action_schema": {
                "type": "object",
                "properties": {
                    "joint_targets": {
                        "type": "array",
                        "items": {"type": "number", "minimum": -math.pi, "maximum": math.pi},
                        "minItems": 2,
                        "maxItems": 2,
                    }
                },
                "required": ["joint_targets"],
                "additionalProperties": False,
            },
            "units": {"joint_targets": "radian", "positions": "meter"},
            "frame": "world XY, shoulder at origin",
            "link_lengths": [0.4, 0.3],
            "control_dt_s": self.frame_skip * self.model.opt.timestep,
            "capabilities": {"visual_observations": self.renderer is not None, "privileged_state": True},
        }

    def reset(self, seed: int) -> Any:
        self.mujoco.mj_resetData(self.model, self.data)
        rng = random.Random(seed)
        self.data.qpos[:] = [rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2)]
        self.data.ctrl[:] = self.data.qpos
        self.data.mocap_pos[0, :2] = self.target
        self.mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        return self.observe()

    def observe(self) -> Any:
        observation = {
            "joint_positions": self.data.qpos.tolist(),
            "joint_velocities": self.data.qvel.tolist(),
            "tip_position": self.data.site_xpos[0, :2].tolist(),
            "target_position": list(self.target),
            "sim_time_s": float(self.data.time),
        }
        if self.renderer is not None:
            from PIL import Image

            self.renderer.update_scene(self.data, camera="top")
            buffer = io.BytesIO()
            Image.fromarray(self.renderer.render()).save(buffer, format="PNG")
            observation["images"] = [{"mime_type": "image/png", "data": base64.b64encode(buffer.getvalue()).decode()}]
        return observation

    def step(self, action: Any) -> Any:
        if not isinstance(action, dict) or set(action) != {"joint_targets"}:
            raise InvalidAction("expected {'joint_targets': [shoulder_rad, elbow_rad]}")
        targets = action["joint_targets"]
        if (
            not isinstance(targets, list)
            or len(targets) != 2
            or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or abs(v) > math.pi
                for v in targets
            )
        ):
            raise InvalidAction("joint_targets must contain two finite angles in [-pi, pi]")
        self.data.ctrl[:] = targets
        self.mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        self.steps += 1
        return self.observe()

    def verify(self) -> RunResult:
        distance = float(self.np.linalg.norm(self.data.site_xpos[0, :2] - self.target))
        speed = float(self.np.linalg.norm(self.data.qvel))
        # Final state only: reaching then moving away cannot keep a stale reward.
        success = distance <= self.threshold and speed <= 0.2 and self.steps > 0
        return RunResult(
            float(success),
            {
                "success": float(success),
                "distance_m": distance,
                "joint_speed_rad_s": speed,
                "sim_steps": float(self.steps),
            },
        )

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
