"""Failures shared by adapters, without simulator-specific exception types."""


class InvalidAction(ValueError):
    """The agent supplied an action outside the environment's contract."""


class BudgetExceeded(RuntimeError):
    """An agent interaction or wall-clock budget was exhausted."""


class AgentError(RuntimeError):
    """A model or agent failed to produce an executable decision."""


class EnvironmentError(RuntimeError):
    """The worker, simulator, or trusted verifier failed."""
