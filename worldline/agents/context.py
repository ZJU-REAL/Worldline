"""Bounded model context without modifying the archived trajectory."""

from __future__ import annotations

import json

from worldline.errors import BudgetExceeded


def bounded_context(messages, max_bytes, measure=None):
    """Retain the initial instructions and complete recent decision groups.

    Never leave an orphan tool reply. If instructions plus the latest turn do
    not fit, fail explicitly; silent truncation inside an observation is unsafe.
    """
    if max_bytes is None:
        return messages, 0
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("context byte budget must be positive")
    groups, prefix = [], []
    for message in messages:
        if message.get("role") == "assistant":
            groups.append([message])
        elif groups:
            groups[-1].append(message)
        else:
            prefix.append(message)

    def size(value):
        return measure(value) if measure else len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode())

    selected = list(groups)
    removed = 0
    sizes = [size(group) for group in groups]
    total = size(prefix) + sum(sizes)
    first = 0
    while total > max_bytes:
        if len(selected) - first <= 1:
            raise BudgetExceeded("instructions and latest interaction exceed model context byte budget")
        total -= sizes[first]
        removed += len(selected[first])
        first += 1
    return prefix + [message for group in selected[first:] for message in group], removed
