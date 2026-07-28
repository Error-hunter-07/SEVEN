import json
import re

from GlobalHelpers.logger import get_logger

log = get_logger(__name__)

# Tolerant tag matcher: the "canonical" format this app is built for is
# <tool_call>{"tool": ..., "arguments": {...}}</tool_call>, but some local
# models (depending on chat template / fine-tune) emit near-miss variants
# using pipe-delimited special tokens instead, e.g. <|tool_call|> ... 
# <tool_call|> — same intent, different wrapper. Matching both here means
# the wrapper gets recognized and stripped either way; the JSON payload
# inside still has to parse cleanly for the call to actually execute.
_TOOL_CALL_TAG = re.compile(
    r"<\|?/?tool_call\|?>(.*?)<\|?/?tool_call\|?>",
    re.DOTALL,
)


def parse_tool_calls(text):
    tool_calls = []

    matches = _TOOL_CALL_TAG.findall(text)

    for m in matches:
        payload = m.strip()
        if not payload:
            continue
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            # The wrapper tag was recognized but the body wasn't valid
            # JSON (e.g. a model emitting "call:tool_name{key:value}"
            # instead of {"tool": "tool_name", "arguments": {...}}).
            # Deliberately NOT attempting to salvage/execute this — a
            # best-effort parse of malformed tool syntax risks calling
            # the wrong tool with the wrong arguments. Logging it so the
            # failure is visible instead of a silent no-op.
            log.warning(
                "Malformed tool-call payload detected and dropped "
                "(not valid JSON): %r",
                payload[:200],
            )
            continue
        if not isinstance(parsed, dict):
            continue
        if "tool" not in parsed:
            continue
        if "arguments" not in parsed or parsed["arguments"] is None:
            parsed["arguments"] = {}
        tool_calls.append(parsed)

    return tool_calls