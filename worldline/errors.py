"""Failures shared by adapters, without simulator-specific exception types."""


class InvalidAction(ValueError):
    """The agent supplied an action outside the environment's contract."""


class BudgetExceeded(RuntimeError):
    """An agent interaction or wall-clock budget was exhausted."""


class AgentError(RuntimeError):
    """A model or agent failed to produce an executable decision."""


class EnvironmentError(RuntimeError):
    """The worker, simulator, or trusted verifier failed."""

    def __init__(self, message: str, *, code: str = "environment.error", retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class ModelAPIError(EnvironmentError):
    """Transport/provider failure; retry classification never relies on message text."""

    def __init__(self, message: str, *, status: int | None = None, retry_after: float | None = None):
        super().__init__(
            message,
            code=f"model.http.{status}" if status else "model.transport",
            retryable=status is None or status in {408, 429, 500, 502, 503, 504},
        )
        self.status = status
        self.retry_after = retry_after


class HTTPRequestError(EnvironmentError):
    """Generic deadline/transport failure for controller and coordinator RPC."""

    def __init__(self, message, *, status=None, retry_after=None):
        super().__init__(
            message,
            code=f"infra.http.{status}" if status else "infra.transport",
            retryable=status is None or status in {408, 429, 500, 502, 503, 504},
        )
        self.status, self.retry_after = status, retry_after
