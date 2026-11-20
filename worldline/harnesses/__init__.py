"""Model-independent, lazy Harness discovery."""

from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Callable


@dataclass(frozen=True)
class HarnessPlugin:
    name: str
    version: str
    backend: str
    protocols: tuple[str, ...]
    factory: Callable | None = None
    harbor_agent: str | None = None
    contract_version: str = "worldline.harness/0.1"


def _tools(**kwargs):
    from worldline.harnesses.native import create_agent

    return create_agent(json_actions=False, **kwargs)


def _json(**kwargs):
    from worldline.harnesses.native import create_agent

    return create_agent(json_actions=True, **kwargs)


def harness_plugins():
    plugins = {
        p.name: p
        for p in (
            HarnessPlugin("tools", "native-tools/0.3", "native", ("chat",), _tools),
            HarnessPlugin("json", "native-json/0.1", "native", ("chat",), _json),
            HarnessPlugin("codex", "harbor-codex/0.23.0", "harbor", ("responses",), harbor_agent="codex"),
        )
    }
    for entry in entry_points(group="worldline.harnesses"):
        plugin = entry.load()()
        if not isinstance(plugin, HarnessPlugin) or plugin.name != entry.name:
            raise ValueError(f"invalid Harness entry point: {entry.name}")
        if plugin.name in plugins:
            raise ValueError(f"duplicate Harness plugin: {plugin.name}")
        plugins[plugin.name] = plugin
    for p in plugins.values():
        if (
            p.contract_version != "worldline.harness/0.1"
            or not p.version
            or p.backend not in {"native", "harbor"}
            or not p.protocols
            or not set(p.protocols) <= {"chat", "responses"}
            or (p.backend == "native" and not callable(p.factory))
            or (p.backend == "harbor" and not p.harbor_agent)
        ):
            raise ValueError(f"incompatible Harness contract: {p.name}")
    return plugins


def get_harness(name):
    if name.startswith("harbor:") and name.removeprefix("harbor:"):
        return HarnessPlugin(name, "harbor-agent/0.23.0", "harbor", ("chat", "responses"), harbor_agent=name[7:])
    try:
        return harness_plugins()[name]
    except KeyError as error:
        raise ValueError(f"unknown Harness: {name}; use worldline harnesses") from error
