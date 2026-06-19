from .parser import parse_tool_calls
from .register import registry

def execute_tool_calls(text):
    tool_calls = parse_tool_calls(text)
    for call in tool_calls:
        tool_name = call.get("tool")
        if not tool_name:
            continue

        tool = registry.get_tool(tool_name)
        if tool is None:
            continue

        arguments = call.get("arguments") or {}
        if not isinstance(arguments, dict):
            arguments = {}

        if tool.parameters:
            normalized_args = {
                name: arguments.get(name)
                for name in tool.parameters
                if name in arguments
            }
            try:
                tool.func(**normalized_args)
            except TypeError:
                try:
                    tool.func()
                except TypeError:
                    continue
        else:
            try:
                tool.func()
            except TypeError:
                continue