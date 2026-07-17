"""
LLMEngine/llm_client.py

CHANGED: After every successful assistant turn, calls
MemoryManagement.semantic_memory.memory_extractor.extract_and_store()
to distil and persist long-term facts from the conversation.

Everything else is unchanged from the original.
"""

import os
import sys
import contextvars
from ToolCalling.register import registry

if __package__ is None or __package__ == "":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Tools.scratchpad_tool import get_scratchpad_memory
import requests

try:
    from .response_parser import parse_response
except ImportError:
    from response_parser import parse_response

import PromptBuilder.prompt_builder as prompt_builder
from Runtime.process_manager import ProcessManager

import ToolCalling.executor as tool_executor

import queue, threading, time
from MemoryManagement.semantic_memory.memory_extractor import extract_and_store, extract_and_store_batch

from Database.chroma_db import wait_for_chroma


from GlobalHelpers.logger import configure_logging, get_logger
from GlobalHelpers.config import settings

# Configure logging once for the process before other components initialize
configure_logging()
log = get_logger(__name__)

# Changes: Added queues extraction instead of immediate direct call to extraction function

_extraction_queue = queue.Queue()
_pending_batch: list[tuple[str, str]] = []
_pending_lock = threading.Lock()
_worker_lock = threading.Lock()
_worker_started = False
_last_extraction_time = 0.0
MIN_EXTRACTION_INTERVAL = 30.0
MAX_BATCH_WAIT = 90.0          # force-flush safety valve
_first_pending_time = None

def _extraction_worker():
    global _last_extraction_time, _first_pending_time
    while True:
        turn = _extraction_queue.get()
        with _pending_lock:
            _pending_batch.append(turn)
            if _first_pending_time is None:
                _first_pending_time = time.time()

        now = time.time()
        cooldown_elapsed = now - _last_extraction_time >= MIN_EXTRACTION_INTERVAL
        waited_too_long = (_first_pending_time is not None
                            and now - _first_pending_time >= MAX_BATCH_WAIT)

        if not (cooldown_elapsed or waited_too_long):
            _extraction_queue.task_done()
            continue

        with _pending_lock:
            batch, _pending_batch[:] = list(_pending_batch), []
            _first_pending_time = None

        extract_and_store_batch(batch)
        _last_extraction_time = time.time()
        _extraction_queue.task_done()

def _start_extraction_worker_with_context() -> None:
    global _worker_started
    with _worker_lock:
        if _worker_started:
            return
        ctx = contextvars.copy_context()
        threading.Thread(
            target=lambda: ctx.run(_extraction_worker),
            daemon=True,
            name="MemoryExtraction",
        ).start()
        _worker_started = True

# ---------------------------------------------------------------------------
# Conversation history
# Persists across turns for KV-cache continuity (see original comments).
# ---------------------------------------------------------------------------
messages: list[dict] = []

process_manager = ProcessManager(
    model_path=settings.llm_model_path,
    llama_cli_path=settings.llm_cli_path,
    mmproj_path=settings.mmproj_path
)

try:
    process_manager.start_for_client()

    # Wait for ChromaDB to finish loading in parallel with llama-server

    if not wait_for_chroma(timeout=120):
        log.warning("ChromaDB did not initialize in time.")

    from SessionManager.session_lifecycle import on_session_start
    session_id = process_manager.session_id
    on_session_start(session_id)
    _start_extraction_worker_with_context()
except Exception:
    log.exception("Failed to start LLM process")


#── BUG FIX 1: max_tokens was 2048 which consumed more space than ──────────
# the model had left after input tokens, causing empty responses.
# Then lowered to 1024 — but that was too tight for code-bearing tool-call
# arguments (confirmed via llama-server logs: 500 errors happened at
# exactly 1024 decoded tokens, mid tool-call JSON, causing truncated/
# invalid argument strings). Raised back up — context window is 16384,
# there's room. Combined with the truncation guard below, this is safe.

MAX_RESPONSE_TOKENS = 2048

# ── BUG FIX 2: conversation history grew unbounded every turn. ─────────────
# After ~4 turns the history alone overflowed the context window.
# Keep only the last N user/assistant pairs. System message is always kept.

MAX_HISTORY_TURNS = 4


def _trim_history(msgs: list[dict]) -> list[dict]:
    """
    Keep system message + last MAX_HISTORY_TURNS of user/assistant pairs.
    Never mutates the original list.
    """
    system = [m for m in msgs if m["role"] == "system"]
    conversation = [m for m in msgs if m["role"] != "system"]
    trimmed = conversation[-(MAX_HISTORY_TURNS * 2):]
    return system + trimmed


def _build_tool_schema(parameters: dict) -> dict:
    """Convert the flat {param_name: description} dict into a valid JSON Schema."""
    if not parameters:
        return {"type": "object", "properties": {}}

    properties = {}
    for name, desc in parameters.items():
        desc_str = str(desc)
        if desc_str.startswith("bool"):
            prop_type = "boolean"
        elif desc_str.startswith("int"):
            prop_type = "integer"
        elif desc_str.startswith("float"):
            prop_type = "number"
        else:
            prop_type = "string"
        properties[name] = {"type": prop_type, "description": desc_str}

    return {
        "type":       "object",
        "properties": properties,
        "required":   list(parameters.keys()),
    }


def request_completion(request_messages: list[dict], use_tools: bool = True) -> dict:
    """
    Returns a dict with:
        text            str   — content field from the model (may be empty)
        native_calls    list  — tool_calls list if finish_reason=tool_calls, else []
        finish_reason   str   — stop | tool_calls | length | null

    use_tools=False skips sending the "tools" schema entirely — use this
    for text-only follow-up requests (e.g. "now give your text reply")
    where forcing tool-call grammar serves no purpose and increases the
    chance of a truncated/invalid generation, especially for code-heavy
    replies. See BUG FIX 3 below.
    """
    payload = {
        "model":       settings.llm_model,
        "messages":    request_messages,
        "temperature": 0.7,
        "max_tokens":  MAX_RESPONSE_TOKENS,
    }
    if use_tools:
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name":        tool.name,
                    "description": tool.description,
                    "parameters":  _build_tool_schema(tool.parameters),
                }
            }
            for tool in registry.list_tools()
        ]

    response = requests.post(
        "http://127.0.0.1:8081/v1/chat/completions",
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    data    = response.json()
    choice  = data.get("choices", [{}])[0]
    message = choice.get("message", {})
    finish_reason = choice.get("finish_reason", "")

    native_calls = message.get("tool_calls") or []

    # ── BUG FIX 3: detect truncated tool-call arguments ─────────────────────
    # Confirmed via llama-server logs: 500 errors occurred at exactly
    # max_tokens decoded, mid tool-call JSON ("unexpected end of input",
    # "missing closing quote"). If the model was cut off while emitting a
    # tool call, its arguments are unreliable — discard rather than risk
    # a malformed downstream call.
    if finish_reason == "length" and native_calls:
        log.warning(
            "Response truncated at max_tokens mid tool-call — "
            "discarding likely-incomplete tool_calls."
        )
        native_calls = []

    return {
        "text":          message.get("content") or "",
        "native_calls":  native_calls,
        "finish_reason": finish_reason,
    }


def ask_llm(query: str) -> str | None:
    fresh_system = {
        "role":    "system",
        "content": prompt_builder.build_prompt(query),
    }
    if messages and messages[0]["role"] == "system":
        messages[0] = fresh_system
    else:
        messages.insert(0, fresh_system)

    messages.append({"role": "user", "content": query})

    try:
        trimmed = _trim_history(messages)
        result  = request_completion(trimmed)

        text          = result["text"]
        native_calls  = result["native_calls"]
        finish_reason = result["finish_reason"]

        # Nothing at all came back
        if not text and not native_calls:
            log.warning("Empty response from model.")
            return "I didn't get a response. Please try again."

        # ── BUG FIX (tool result propagation): execute tools and capture
        # what each one actually returned, keyed by tool_call_id, so the
        # follow-up "give your text reply" turn can see real results
        # instead of a hardcoded "Done." placeholder. Previously the model
        # only saw real tool output on the NEXT ask_llm() call (via the
        # scratchpad-rebuilt system prompt), which is why answers appeared
        # to "read the fetch" a turn late.
        tool_results = tool_executor.execute_tool_calls(
            text=text,
            native_tool_calls=native_calls,
        )

        # Build what goes into history
        # When finish_reason=tool_calls, content is empty — store it properly
        history_message = {
            "role":    "assistant",
            "content": text or None,
        }
        if native_calls:
            history_message["tool_calls"] = native_calls

        # Get user-facing text
        parsed_response = parse_response(text) if text else ""

        # If there's no text (pure tool-call reply), ask for a follow-up text reply
        if not parsed_response.strip():
            follow_messages = _trim_history(messages) + [
                history_message,
                *[
                    {
                        "role":         "tool",
                        "tool_call_id": tc.get("id", ""),
                        "content":      tool_results.get(tc.get("id", ""), "Done."),
                    }
                    for tc in native_calls
                ],
                {"role": "user", "content": "Now give your text reply."},
            ]
            try:
                # BUG FIX: use_tools=False — this is a text-only follow-up,
                # forcing tool-call grammar here serves no purpose and was
                # the second half of the 500-error chain seen in production
                # logs (a truncated tool call followed by a second forced
                # tool-call attempt in the follow-up).
                follow_result = request_completion(follow_messages, use_tools=False)
                parsed_response = parse_response(follow_result["text"])
            except requests.exceptions.HTTPError as e:
                log.error("Follow-up completion failed: %s", e, exc_info=True)
                parsed_response = text or "I ran into an issue finishing that response — could you try rephrasing or asking again?"

        if not parsed_response.strip():
            parsed_response = "Done."

        messages.append(history_message)
# ==========================================================
        # FIX:
        # Build a summary of any native tool calls so memory extraction
        # still has assistant context even when text == "".
        # ==========================================================
        tool_summary = ""
        if native_calls:
            parts = [
                f"[Called {tc['function']['name']}]"
                for tc in native_calls
            ]
            tool_summary = " ".join(parts)

        full_assistant_activity = f"{text} {tool_summary}".strip()

        # _maybe_extract_memory(query, full_assistant_activity)

        # Instead of calling extract_and_store directly:
        _extraction_queue.put((query, full_assistant_activity))
        # ==========================================================


        # _maybe_extract_memory(query, text)

        log.debug(get_scratchpad_memory())
        return parsed_response

    except requests.exceptions.ConnectionError:
        log.error("Could not connect to local LLM server at :8081.")
    except requests.exceptions.Timeout:
        log.error("Model timed out.")
    except requests.exceptions.RequestException as e:
        log.error("Request failed: %s", e, exc_info=True)
    except Exception as e:
        log.exception("Unexpected error during ask_llm")




def _maybe_extract_memory(user_message: str, assistant_activity: str) -> None:
    # Guard on USER message length, not assistant reply
    if len(user_message.split()) < 8:
        return
    # No need to check assistant_activity length anymore —
    # we always have something now (either text or tool summary)
    try:
        extract_and_store(user_message=user_message, assistant_reply=assistant_activity)
    except Exception as e:
        log.exception("Memory extraction error (non-fatal)")


if __name__ == "__main__":
    while True:
        user_query = input("You: ")
        from SessionManager.session_lifecycle import on_session_end
        if user_query.strip().lower() == "/stop":
            on_session_end(process_manager.session_id)
            process_manager.stop_from_cli()
            break

        answer = ask_llm(user_query)

        print("\nAssistant:")
        print(answer)