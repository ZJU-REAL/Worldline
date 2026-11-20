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
    return {**runtime, "settings": comparison_settings(runtime["settings"])}


def comparison_settings(settings):
    return {key: value for key, value in settings.items() if key not in {"model", "model_source", "model_revision"}}


def factory_configuration(factory):
    candidate = getattr(factory, "func", factory)
    try:
        source = inspect.getsourcefile(candidate)
    except TypeError:
        source = None
    return {
        "name": f"{getattr(candidate, '__module__', '')}.{getattr(candidate, '__qualname__', type(candidate).__qualname__)}",
        "code_sha256": file_sha256(Path(source)) if source and Path(source).is_file() else None,
    }
