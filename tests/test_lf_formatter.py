from __future__ import annotations

import json
import types
import unittest
from collections import namedtuple
from unittest.mock import patch

from worldline.learning.llama_factory_entry import register_template


class FormatterTest(unittest.TestCase):
    def test_prose_is_inside_native_assistant_slots_and_base_template_is_unchanged(self):
        class StringFormatter:
            def apply(self, **kwargs):
                return [slot.replace("{{content}}", kwargs["content"]) for slot in self.slots]

        class FunctionFormatter(StringFormatter):
            def __init__(self, slots, tool_format):
                self.slots, self.tool_format = slots, tool_format
                self.tool_utils = types.SimpleNamespace(
                    function_formatter=lambda calls: "<tool>" + calls[0].name + ":" + calls[0].arguments + "</tool>"
                )

        formatter = types.ModuleType("llamafactory.data.formatter")
        formatter.StringFormatter, formatter.FunctionFormatter = StringFormatter, FunctionFormatter
        formatter.FunctionCall = namedtuple("FunctionCall", "name arguments")
        templates = types.ModuleType("llamafactory.data.template")
        base = FunctionFormatter(["assistant:{{content}}:eos"], "native")
        templates.TEMPLATES = {"native": types.SimpleNamespace(format_function=base)}
        with patch.dict(
            "sys.modules", {"llamafactory.data.formatter": formatter, "llamafactory.data.template": templates}
        ):
            alias = register_template("native")
            value = json.dumps([{"name": "act", "arguments": {"x": 1}, "_worldline_assistant_prose": "plan"}])
            result = templates.TEMPLATES[alias].format_function.apply(content=value)
            self.assertEqual(result, ['assistant:plan\n<tool>act:{"x": 1}</tool>:eos'])
            self.assertIs(templates.TEMPLATES["native"].format_function, base)
