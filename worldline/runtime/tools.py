"""Public tool dispatch, independent of model APIs and trainers."""

from worldline.errors import InvalidAction


def public_spec(spec):
    return {**spec, "capabilities": {**spec.get("capabilities", {}), "snapshot": False, "fork": False}}


def execute_tool(name, arguments, sandbox):
    if name == "act" and set(arguments) == {"action"}:
        return sandbox.act(arguments["action"]), False
    if name == "observe" and not arguments:
        return sandbox.observe(), False
    if name == "reset" and not arguments:
        return sandbox.reset(), False
    if name == "finish" and not arguments:
        return {"submitted": True}, True
    raise InvalidAction("unknown tool or invalid arguments")
