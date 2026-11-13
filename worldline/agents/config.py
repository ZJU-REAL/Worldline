"""Effective harness identity for reporting and comparable evaluations."""

from __future__ import annotations

import inspect
from pathlib import Path
from worldline.storage import file_sha256


def runtime_configuration(agent):
    settings = getattr(agent, "worldline_config", {})
    if callable(settings):
        settings = settings()
    if not isinstance(settings, dict):
        raise ValueError("agent worldline_config must be a JSON object")
    module = inspect.getmodule(type(agent))
    path = Path(module.__file__) if module and getattr(module, "__file__", None) else None
    return {
        "class": f"{type(agent).__module__}.{type(agent).__qualname__}",
        "code_sha256": file_sha256(path) if path and path.is_file() else None,
        "settings": settings,
    }


def comparison_configuration(runtime):
    # Model names are reported separately so models using the same harness and
    # budgets can still be compared and paired on a common configuration.
    return {**runtime, "settings": {key: value for key, value in runtime["settings"].items() if key != "model"}}
