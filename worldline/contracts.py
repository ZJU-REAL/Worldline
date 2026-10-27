"""Small contracts shared by RUA, CUA, and future sandbox adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class SandboxCapabilities:
    """Optional behavior exposed by a sandbox backend."""

    snapshot: bool = False
    fork: bool = False
    realtime: bool = False
    deterministic_replay: bool = False
    visual_observations: bool = False
    privileged_state: bool = False


@dataclass(frozen=True, slots=True)
class Task:
    """A versioned task independent of a particular agent or trainer."""

    id: str
    instruction: str
    environment: str
    version: str
    split: str = "unspecified"
    seed: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RunResult:
    """Trusted outcome returned by a verifier."""

    reward: float
    metrics: Mapping[str, float] = field(default_factory=dict)
    evidence: tuple[Path, ...] = ()
    infra_error: str | None = None


@runtime_checkable
class Sandbox(Protocol):
    """Lifecycle boundary implemented by every environment adapter."""

    @property
    def capabilities(self) -> SandboxCapabilities: ...

    def start(self, task: Task) -> Any: ...

    def observe(self) -> Any: ...

    def act(self, action: Any) -> Any: ...

    def reset(self) -> Any: ...

    def close(self) -> None: ...


@runtime_checkable
class Agent(Protocol):
    """An agent that operates inside a prepared sandbox."""

    def run(self, task: Task, sandbox: Sandbox) -> Any: ...


@runtime_checkable
class Verifier(Protocol):
    """A trusted evaluator isolated from the agent."""

    def verify(self, task: Task, run_dir: Path) -> RunResult: ...
