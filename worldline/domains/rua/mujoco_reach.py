"""A real, CPU-capable MuJoCo integration fixture, not a benchmark score."""

from __future__ import annotations

import base64
import hashlib
import io
import math
import random
import json
import os
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
        if config.get("render_backend") is not None:
            if config["render_backend"] not in {"egl", "osmesa", "glfw"}:
                raise ValueError("render_backend must be egl, osmesa or glfw")
            os.environ["MUJOCO_GL"] = config["render_backend"]
        device = os.environ.get("WORLDLINE_GPU_DEVICE")
        if config.get("render") and device is not None:
            selected = config.get("egl_device_id", int(device) if device.isdecimal() else None)
            if type(selected) is not int or selected < 0:
                raise ValueError("GPU rendering requires a numeric device or explicit egl_device_id")
            if os.environ.get("MUJOCO_GL", "egl") != "egl":
                raise ValueError("GPU-rendered fixture requires MUJOCO_GL=egl in its fresh worker")
            os.environ["MUJOCO_GL"] = "egl"
            os.environ["MUJOCO_EGL_DEVICE_ID"] = str(selected)
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
        self.render_backend = mujoco.gl_context.GLContext.__module__ if self.renderer else "disabled"
        self.render_engine = "disabled"
        if self.renderer:
            from OpenGL import GL

            self.render_engine = (GL.glGetString(GL.GL_RENDERER) or b"unknown").decode()
            if device is not None and any(
                name in self.render_engine.lower() for name in ("llvmpipe", "softpipe", "software")
            ):
                self.renderer.close()
                raise ValueError(
                    "GPU lease selected a software renderer; use CPU rendering or configure the physical EGL device"
                )
        self.steps = self.total_steps = 0
        self.inherited_steps = 0
        self.checkpoint_config = hashlib.sha256(
            json.dumps(
                {
                    "target": self.target,
                    "threshold": self.threshold,
                    "frame_skip": self.frame_skip,
                    "render": self.renderer is not None,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        self.reset(task.seed)

    def spec(self) -> Mapping[str, Any]:
        return {
            "protocol": "worldline.rua/0.1",
            "robot": "planar-two-link-arm",
            "simulator_version": self.mujoco.__version__,
            "render_backend": self.render_backend,
            "render_engine": self.render_engine,
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
            "capabilities": {
                "visual_observations": self.renderer is not None,
                "privileged_state": True,
                "snapshot": True,
                "fork": True,
            },
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
        self.total_steps += self.frame_skip
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
                "sim_steps": float(self.steps * self.frame_skip),
                "sim_steps_total": float(self.total_steps),
                "sim_steps_executed": float(self.total_steps - self.inherited_steps),
            },
        )

    def checkpoint(self):
        state_type = self.mujoco.mjtState.mjSTATE_INTEGRATION
        state = self.np.empty(self.mujoco.mj_stateSize(self.model, state_type))
        self.mujoco.mj_getState(self.model, self.data, state, state_type)
        # Preserve the presentation state too: mj_step's final integration can
        # leave cached transforms at a different phase from qpos. Rendering and
        # the next public observation must match the source checkpoint exactly.
        fields = ("xpos", "xquat", "xmat", "geom_xpos", "geom_xmat", "site_xpos", "site_xmat", "cam_xpos", "cam_xmat")
        cached = {
            name: {"shape": list(getattr(self.data, name).shape), "values": getattr(self.data, name).ravel().tolist()}
            for name in fields
        }
        return {
            "schema_version": "worldline.mujoco-checkpoint/0.1",
            "environment_version": self.version(),
            "mujoco_version": self.mujoco.__version__,
            "config": self.checkpoint_config,
            "state": state.tolist(),
            "steps": self.steps,
            "cumulative_steps": self.total_steps,
            "cached_transforms": cached,
        }

    def execution_info(self):
        return {
            "gpu_device": os.environ.get("WORLDLINE_GPU_DEVICE"),
            "render_backend": self.render_backend,
            "render_engine": self.render_engine,
            "egl_device_id": os.environ.get("MUJOCO_EGL_DEVICE_ID") if self.renderer else None,
        }

    def restore(self, checkpoint):
        if (
            checkpoint.get("schema_version") != "worldline.mujoco-checkpoint/0.1"
            or checkpoint.get("environment_version") != self.version()
            or checkpoint.get("mujoco_version") != self.mujoco.__version__
            or checkpoint.get("config") != self.checkpoint_config
        ):
            raise ValueError("checkpoint environment/configuration mismatch")
        state_type = self.mujoco.mjtState.mjSTATE_INTEGRATION
        state = self.np.asarray(checkpoint["state"], dtype=float)
        if state.shape != (self.mujoco.mj_stateSize(self.model, state_type),) or not self.np.isfinite(state).all():
            raise ValueError("invalid checkpoint integration state")
        if type(checkpoint["steps"]) is not int or checkpoint["steps"] < 0:
            raise ValueError("invalid checkpoint step counter")
        transforms = {}
        expected = ("xpos", "xquat", "xmat", "geom_xpos", "geom_xmat", "site_xpos", "site_xmat", "cam_xpos", "cam_xmat")
        for name in expected:
            value = checkpoint["cached_transforms"][name]
            shape = tuple(value["shape"])
            if shape != getattr(self.data, name).shape:
                raise ValueError("checkpoint transform shape mismatch")
            array = self.np.asarray(value["values"], dtype=float).reshape(shape)
            if not self.np.isfinite(array).all():
                raise ValueError("invalid checkpoint transform")
            transforms[name] = array
        self.mujoco.mj_setState(self.model, self.data, state, state_type)
        self.mujoco.mj_forward(self.model, self.data)
        # Restore integration inputs/warm start after rebuilding derived fields.
        self.mujoco.mj_setState(self.model, self.data, state, state_type)
        for name, array in transforms.items():
            getattr(self.data, name)[:] = array
        self.steps = checkpoint["steps"]
        # Cumulative compute is deliberately never rewound.
        return self.observe()

    def fork_restore(self, checkpoint):
        cumulative = checkpoint["cumulative_steps"]
        if type(cumulative) is not int or cumulative < checkpoint["steps"] * self.frame_skip:
            raise ValueError("invalid fork compute counter")
        observation = self.restore(checkpoint)
        self.inherited_steps += max(0, cumulative - self.total_steps)
        self.total_steps = max(self.total_steps, cumulative)
        return observation

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
