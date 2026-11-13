"""Live native rollout/reward bridge for independent training loops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Any
from pathlib import Path
import os
import math
from worldline.runtime.resources import ResourcePool

from worldline.runtime.interactive import InteractiveSession
from worldline.tasks import load_native_task

_RESOURCE_PID = os.getpid()
_DEFAULT_RESOURCES = ResourcePool(os.cpu_count() or 1, ())


def default_resources():
    global _RESOURCE_PID, _DEFAULT_RESOURCES
    if _RESOURCE_PID != os.getpid():
        _RESOURCE_PID = os.getpid()
        _DEFAULT_RESOURCES = ResourcePool(os.cpu_count() or 1, ())
    return _DEFAULT_RESOURCES


@dataclass(frozen=True)
class TrainingOutcome:
    run_id: str
    run_dir: str
    reward: float | None
    valid: bool
    outcome: str
    task_id: str
    task_version: str
    seed: int
    metrics: Mapping[str, Any]
    error: str | None = None


class NativeTrainingSession(InteractiveSession):
    """A fresh train-only environment and a verifier score of newly generated actions.

    Tool replies are public observations. Reward is available only to the trusted
    trainer at finalization. Infrastructure errors have reward=None and valid=False;
    callers must discard them before computing advantages or supervising samples.
    """

    def __init__(self, task_dir: Path, *, resources: ResourcePool | None = None, **options):
        definition = load_native_task(task_dir)
        if definition.task.split != "train":
            raise ValueError("live training requires an explicitly declared train task")
        self.resources = resources or default_resources()
        requested = definition.task.metadata.get("resources", {})
        self.cpus = max(1, math.ceil(requested.get("min_cpus", 1)))
        self.gpu_required = bool(requested.get("gpu_required", False))
        if self.cpus > self.resources.capacity or self.gpu_required and not self.resources.gpu_capacity:
            raise ValueError("training task exceeds configured CPU/GPU pool capacity")
        self.device, self.leased = None, False
        super().__init__(task_dir, **options)

    def start(self):
        with self.lock:
            if self.environment is not None or self.result is not None:
                raise RuntimeError("session was already started or closed")
            self.device = self.resources.acquire(self.cpus, self.gpu_required)
            self.leased = True
            if self.device is not None:
                self.worker_env = {
                    **(self.worker_env or {}),
                    "CUDA_VISIBLE_DEVICES": str(self.device),
                    "WORLDLINE_GPU_DEVICE": str(self.device),
                }
            return super().start()

    def _publish(self, *args, **kwargs):
        try:
            return super()._publish(*args, **kwargs)
        finally:
            if self.leased and (self.environment is None or self.environment.process.poll() is not None):
                self.resources.release(self.cpus, self.device)
                self.leased = False

    def training_outcome(self) -> TrainingOutcome:
        result = self.finish()
        valid = result.outcome in {"success", "failure"}
        return TrainingOutcome(
            result.trial_id,
            str(self.run_dir),
            result.reward if valid else None,
            valid,
            result.outcome,
            self.task.id,
            self.task.version,
            self.task.seed,
            result.metrics,
            result.error,
        )
