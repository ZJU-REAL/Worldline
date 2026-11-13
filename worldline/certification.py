"""Executable environment contract certification, without benchmark score claims."""

import json
import tempfile
from pathlib import Path

from worldline.runtime.process import ProcessEnvironment
from worldline.runtime.verdict import checked_verdict
from worldline.tasks import load_native_task


def certify_environment(task_dir):
    definition = load_native_task(task_dir)
    with tempfile.TemporaryDirectory(prefix="worldline-certification-") as temp:
        root = Path(temp)
        environment = ProcessEnvironment(definition, root)
        try:
            initial = environment.start()
            json.dumps(initial, allow_nan=False)
            spec = environment.spec
            json.dumps(spec, allow_nan=False)
            schema = spec.get("action_schema")
            if not isinstance(schema, dict) or not schema.get("type"):
                raise ValueError("environment must declare a JSON action schema")
            capabilities = spec.get("capabilities", {})
            if any(type(value) is not bool for value in capabilities.values()):
                raise ValueError("capabilities must be booleans")
            checkpoint_tested = False
            if capabilities.get("snapshot"):
                checkpoint = environment.checkpoint()
                restored = environment.restore(checkpoint)
                if restored != initial:
                    raise ValueError("checkpoint roundtrip changed the observation")
                checkpoint_tested = True
            verdict = environment.verify(definition.task, root)
            outcome, _, _ = checked_verdict(verdict)
            return {
                "schema_version": "worldline.certification/0.1",
                "task_id": definition.task.id,
                "task_version": definition.task.version,
                "environment": definition.task.environment,
                "passed": True,
                "checkpoint_tested": checkpoint_tested,
                "initial_outcome": outcome,
                "execution": environment.execution,
                "benchmark_accuracy_tested": False,
            }
        finally:
            environment.close()
