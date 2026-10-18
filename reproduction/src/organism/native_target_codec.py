"""Versioned correction of scalar literals in Granite assistant tool targets.

The upstream template renders scalar None/True/False with Python spelling.
llama.cpp b9850's generated XML tool parser requires JSON for nonstring
arguments. Only the supervised action is transformed. Serving prompts,
observations, source argument values, and the pinned upstream template remain
unchanged. The caller must independently round-trip targets through that parser.
"""

import json
import re


UPSTREAM = "upstream-native-v1"
JSON_SCALARS = "granite-json-scalar-targets-v1"
PROFILES = (UPSTREAM, JSON_SCALARS)
_FUNCTION = re.compile(r"<function=([^>\n]+)>(.*?)</function>", re.DOTALL)
_PARAMETER = re.compile(r"<parameter=([^>\n]+)>(.*?)</parameter>", re.DOTALL)


def json_scalar_target(text, completion):
    """Preserve native syntax while correcting typed null and boolean scalars."""
    if not isinstance(text, str) or completion.get("role") != "assistant":
        raise ValueError("Expected a native assistant target")
    calls = completion.get("tool_calls") or []
    if not calls:
        return text
    functions = list(_FUNCTION.finditer(text))
    if len(functions) != len(calls):
        raise ValueError("Native target function inventory differs from source actions")
    changes = []
    for function, call in zip(functions, calls):
        expected = call["function"]
        arguments = expected["arguments"]
        if function[1] != expected["name"] or not isinstance(arguments, dict):
            raise ValueError("Native target function name or source arguments differ")
        parameters = list(_PARAMETER.finditer(function[2]))
        names = [parameter[1] for parameter in parameters]
        if len(names) != len(set(names)) or set(names) != set(arguments):
            raise ValueError("Native target argument inventory differs from source actions")
        for parameter in parameters:
            value = arguments[parameter[1]]
            if value is not None and type(value) is not bool:
                continue
            expected_python = "None" if value is None else str(value)
            expected_json = json.dumps(value)
            raw = parameter[2]
            if raw.strip() not in (expected_python, expected_json):
                raise ValueError("Native target scalar differs from source argument")
            # Preserve every surrounding byte, including template whitespace.
            start = len(raw) - len(raw.lstrip())
            end = len(raw.rstrip())
            offset = function.start(2) + parameter.start(2)
            changes.append((offset + start, offset + end, expected_json))
    for start, end, replacement in reversed(changes):
        text = text[:start] + replacement + text[end:]
    return text
