import json
from .parser import parse_tool_calls
from .register import registry
from GlobalHelpers.logger import get_logger

log = get_logger(__name__)

def execute_tool_calls(text: str = "", native_tool_calls: list = None) -> dict:
    """Returns {tool_call_id: result_string} for native calls, so callers
    can feed real tool output back to the model instead of a placeholder."""
    results = {}
    tag_calls = parse_tool_calls(text) if text else []

    api_calls = []
    for tc in (native_tool_calls or []):
        fn = tc.get("function", {})
        name = fn.get("name", "")
        args_raw = fn.get("arguments", "{}")
        call_id = tc.get("id", "")
        if not name:
            continue
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        except json.JSONDecodeError:
            args = {}
        api_calls.append({"tool": name, "arguments": args, "call_id": call_id})

    for call in tag_calls:
        _run_call(call)
    for call in api_calls:
        result_str = _run_call(call)
        if call.get("call_id"):
            results[call["call_id"]] = result_str

    return results


def _run_call(call: dict) -> str:
    tool_name = call.get("tool")
    if not tool_name:
        return "No tool name given."

    tool = registry.get_tool(tool_name)
    if tool is None:
        log.warning("Unknown tool: '%s'", tool_name)
        return f"Unknown tool: {tool_name}"

    arguments = call.get("arguments") or {}
    if not isinstance(arguments, dict):
        arguments = {}

    normalized_args = {
        name: arguments.get(name)
        for name in (tool.parameters or {})
        if name in arguments
    } if tool.parameters else {}

    try:
        result = tool.func(**normalized_args)
    except TypeError:
        try:
            result = tool.func()
        except TypeError:
            log.error("Failed to call '%s' with args %s", tool_name, normalized_args)
            return f"Failed to call {tool_name}."

    if result is None:
        return "Done."
    if isinstance(result, (dict, list)):
        return json.dumps(result, default=str)[:2000]
    return str(result)[:2000]