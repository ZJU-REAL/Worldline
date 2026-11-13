"""Trusted verdict validation shared by evaluation and live training sessions."""

import math


def checked_verdict(verdict):
    if verdict.infra_error:
        raise RuntimeError(verdict.infra_error)
    if type(verdict.reward) not in {int, float} or not math.isfinite(verdict.reward):
        raise RuntimeError("verifier returned a non-finite or invalid reward")
    if any(type(value) not in {int, float} or not math.isfinite(value) for value in verdict.metrics.values()):
        raise RuntimeError("verifier returned invalid metrics")
    success = verdict.metrics.get("success", float(verdict.reward == 1))
    if success not in {0, 1}:
        raise RuntimeError("verifier success must be zero or one")
    return "success" if success == 1 else "failure", float(verdict.reward), dict(verdict.metrics)
