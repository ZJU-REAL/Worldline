"""Standalone LLaMA-Factory formatter entry preserving prose plus tool calls."""

from __future__ import annotations

import copy
import json
import sys


def prose_and_calls(content):
    calls = json.loads(content)
    if not isinstance(calls, list) or not calls:
        raise ValueError("function supervision must contain a nonempty call list")
    prose = calls[0].get("_worldline_assistant_prose", "")
    if not isinstance(prose, str):
        raise ValueError("assistant prose must be text")
    return prose, [{"name": call["name"], "arguments": call["arguments"]} for call in calls]


def register_template(name):
    from llamafactory.data.template import TEMPLATES
    from llamafactory.data.formatter import FunctionFormatter, StringFormatter, FunctionCall

    base = name.removeprefix("worldline__")
    original = copy.deepcopy(TEMPLATES[base])
    function = original.format_function
    if not isinstance(function, FunctionFormatter):
        raise ValueError("this template does not expose the supported FunctionFormatter contract")

    class ProseFunctionFormatter(FunctionFormatter):
        def apply(self, **kwargs):
            prose, calls = prose_and_calls(kwargs["content"])
            functions = [
                FunctionCall(call["name"], json.dumps(call["arguments"], ensure_ascii=False)) for call in calls
            ]
            text = self.tool_utils.function_formatter(functions)
            # StringFormatter places content inside the source template's
            # assistant header/EOS slots, preserving native tool syntax.
            return StringFormatter.apply(self, content=(prose + "\n" if prose else "") + text)

    original.format_function = ProseFunctionFormatter(
        slots=copy.deepcopy(function.slots), tool_format=function.tool_format
    )
    alias = "worldline__" + base
    TEMPLATES[alias] = original
    return alias


def main():
    import yaml
    from llamafactory.train.tuner import run_exp

    with open(sys.argv[1]) as stream:
        config = yaml.safe_load(stream)
    config["template"] = register_template(config["template"])
    run_exp(args=config)


if __name__ == "__main__":
    main()
