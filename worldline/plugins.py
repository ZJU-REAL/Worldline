"""Lazy environment registration for RUA, CUA, and third-party domains."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Any, Callable, Mapping

from worldline.contracts import Environment, Task


@dataclass(frozen=True)
class EnvironmentPlugin:
    name: str
    domain: str
    version: str
    factory: Callable[[Task, Mapping[str, Any]], Environment]


def environment_plugins() -> dict[str, EnvironmentPlugin]:
    # Importing these modules must not import their optional simulator libraries.
    from worldline.domains.rua import plugins

    registered = {plugin.name: plugin for plugin in plugins()}
    for entry in entry_points(group="worldline.environments"):
        plugin = entry.load()()
        if not isinstance(plugin, EnvironmentPlugin) or plugin.name != entry.name:
            raise ValueError(f"invalid environment entry point: {entry.name}")
        if plugin.name in registered:
            raise ValueError(f"duplicate environment plugin: {plugin.name}")
        registered[plugin.name] = plugin
    return registered


def get_environment_plugin(name: str) -> EnvironmentPlugin:
    try:
        return environment_plugins()[name]
    except KeyError as error:
        raise ValueError(f"unknown environment plugin: {name}") from error
