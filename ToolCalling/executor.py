import json
from .parser import parse_tool_calls
from .register import registry


def execute_tool_calls(text: str = "", native_tool_calls: list = None) -> None:
    """
    Handles both tool call formats:

    Format 1 — tag format (model replied in plain text with embedded tags):
        <tool_call>
        {"tool": "update_scratchpad_state", "arguments": {...}}
        </tool_call>
        Parsed from `text` by the existing parser.

    Format 2 — native API format (finish_reason=tool_calls, content is empty):
        message["tool_calls"] = [
            {"type": "function", "function": {"name": "...", "arguments": "..."}, "id": "..."}
        ]
        Passed directly as `native_tool_calls`.

    Either or both can be present in a single turn. Both are executed.
    """

    # ── Format 1: parse <tool_call> tags from text ───────────────────────────
    tag_calls = parse_tool_calls(text) if text else []

    # ── Format 2: parse native API tool_calls list ───────────────────────────
    api_calls = []
    for tc in (native_tool_calls or []):
        fn       = tc.get("function", {})
        name     = fn.get("name", "")
        args_raw = fn.get("arguments", "{}")
        if not name:
            continue
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        except json.JSONDecodeError:
            args = {}
        api_calls.append({"tool": name, "arguments": args})

    # ── Execute all calls in order: tag calls first, then API calls ──────────
    for call in tag_calls + api_calls:
        _run_call(call)


def _run_call(call: dict) -> None:
    tool_name = call.get("tool")
    if not tool_name:
        return

    tool = registry.get_tool(tool_name)
    if tool is None:
        print(f"[Executor] Unknown tool: '{tool_name}'")
        return

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
                print(f"[Executor] Failed to call '{tool_name}' with args {normalized_args}")
    else:
        try:
            tool.func()
        except TypeError:
            print(f"[Executor] Failed to call '{tool_name}' (no params)")