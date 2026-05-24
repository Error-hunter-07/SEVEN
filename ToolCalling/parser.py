

import json
import re


def parse_tool_calls(text):
    tool_calls = []

    matches = re.findall(
        r"<tool_call>(.*?)</tool_call>",
        text,
        re.DOTALL,
    )

    for m in matches:
        payload = m.strip()
        if not payload:
            continue
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        if "tool" not in parsed:
            continue
        if "arguments" not in parsed or parsed["arguments"] is None:
            parsed["arguments"] = {}
        tool_calls.append(parsed)

    return tool_calls