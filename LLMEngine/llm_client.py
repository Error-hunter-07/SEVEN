"""
LLMEngine/llm_client.py

CHANGED: After every successful assistant turn, calls
MemoryManagement.semantic_memory.memory_extractor.extract_and_store()
to distil and persist long-term facts from the conversation.

Everything else is unchanged from the original.
"""

import os
import sys
from ToolCalling.register import registry

if __package__ is None or __package__ == "":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Tools.scratchpad_tool import get_scratchpad_memory
import requests

try:
    from .response_parser import parse_response
except ImportError:
    from response_parser import parse_response

from dotenv import load_dotenv
import PromptBuilder.prompt_builder as prompt_builder
from Runtime.process_manager import ProcessManager

import ToolCalling.executor as tool_executor

# CHANGED: import memory extractor for post-turn semantic memory extraction
from MemoryManagement.semantic_memory.memory_extractor import extract_and_store

# llm_client.py — top of file, with your other imports
import queue, threading, time
from MemoryManagement.semantic_memory.memory_extractor import extract_and_store

load_dotenv()

# Changes: Added queues extraction instead of immediate direct call to extraction function

_extraction_queue = queue.Queue()
_last_extraction_time = 0.0
MIN_EXTRACTION_INTERVAL = 30.0

def _extraction_worker():
    global _last_extraction_time
    while True:
        user_msg, assistant_reply = _extraction_queue.get()
        now = time.time()
        if now - _last_extraction_time < MIN_EXTRACTION_INTERVAL:
            _extraction_queue.task_done()
            continue
        extract_and_store(user_msg, assistant_reply)
        _last_extraction_time = time.time()
        _extraction_queue.task_done()

threading.Thread(target=_extraction_worker, daemon=True).start()
# ---------------------------------------------------------------------------
# Conversation history
# Persists across turns for KV-cache continuity (see original comments).
# ---------------------------------------------------------------------------
messages: list[dict] = []

process_manager = ProcessManager(
    model_path=os.getenv("LLM_MODEL_PATH"),
    llama_cli_path=os.getenv("LLM_CLI_PATH"),
    mmproj_path=os.getenv("MMPROJ_PATH")
)

try:
    process_manager.start_for_client()
    from SessionManager.session_lifecycle import on_session_start
    session_id = process_manager.session_id
    on_session_start(session_id)
except Exception as e:
    print(f"[ERROR] Failed to start LLM process: {e}")


#── BUG FIX 1: max_tokens was 2048 which consumed more space than ──────────
# the model had left after input tokens, causing empty responses.
# Set to 1024 — enough for any normal reply, leaves room for input.

MAX_RESPONSE_TOKENS = 1024
 
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


# TEMPORARY DEBUG VERSION — revert after finding the issue
# Replace request_completion with this to see raw server output:
 
def request_completion(request_messages: list[dict]) -> dict:
    """
    Returns a dict with:
        text            str   — content field from the model (may be empty)
        native_calls    list  — tool_calls list if finish_reason=tool_calls, else []
        finish_reason   str   — stop | tool_calls | length | null
    """
    response = requests.post(
        "http://127.0.0.1:8081/v1/chat/completions",
        json={
            "model":       os.getenv("LLM_MODEL"),
            "messages":    request_messages,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name":        tool.name,
                        "description": tool.description,
                        "parameters":  _build_tool_schema(tool.parameters),
                    }
                }
                for tool in registry.list_tools()
            ],
            "temperature": 0.7,
            "max_tokens":  MAX_RESPONSE_TOKENS,
        },
        timeout=120,
    )
    response.raise_for_status()
    data    = response.json()
    choice  = data.get("choices", [{}])[0]
    message = choice.get("message", {})
 
    return {
        "text":          message.get("content") or "",
        "native_calls":  message.get("tool_calls") or [],
        "finish_reason": choice.get("finish_reason", ""),
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
            print("[llm_client] Empty response from model.")
            return "I didn't get a response. Please try again."
 
        # Execute tools — executor now handles both formats
        tool_executor.execute_tool_calls(
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
                        "content":      "Done.",
                    }
                    for tc in native_calls
                ],
                {"role": "user", "content": "Now give your text reply."},
            ]
            follow_result = request_completion(follow_messages)
            parsed_response = parse_response(follow_result["text"])
 
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
 
        print(get_scratchpad_memory())
        return parsed_response
 
    except requests.exceptions.ConnectionError:
        print("[ERROR] Could not connect to local LLM server at :8081.")
    except requests.exceptions.Timeout:
        print("[ERROR] Model timed out.")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Request failed: {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {type(e).__name__}: {e}")
 



def _maybe_extract_memory(user_message: str, assistant_activity: str) -> None:
    # Guard on USER message length, not assistant reply
    if len(user_message.split()) < 8:
        return
    # No need to check assistant_activity length anymore —
    # we always have something now (either text or tool summary)
    try:
        extract_and_store(user_message=user_message, assistant_reply=assistant_activity)
    except Exception as e:
        print(f"[llm_client] Memory extraction error (non-fatal): {e}")


if __name__ == "__main__":
    while True:
        user_query = input("You: ")
        from SessionManager.session_lifecycle import on_session_end
        if user_query.strip().lower() == "/stop":
            on_session_end(get_session_id())
            process_manager.stop_from_cli()
            break

        answer = ask_llm(user_query)

        print("\nAssistant:")
        print(answer)
