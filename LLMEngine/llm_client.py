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

load_dotenv()


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
except Exception as e:
    print(f"[ERROR] Failed to start LLM process: {e}")


# def _build_system_message() -> dict:
#     """Build the system message dict with a freshly compiled prompt."""
#     return {
#         "role":    "system",
#         "content": prompt_builder.build_prompt("")
#     }


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


def request_completion(request_messages):
    response = requests.post(
        "http://127.0.0.1:8081/v1/chat/completions",
        json={
            "model":      os.getenv("LLM_MODEL"),
            "messages":   request_messages,
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
            "max_tokens":  MAX_RESPONSE_TOKENS, #Fixed this from 2048 to 1024
        },
        timeout=120
    )
    response.raise_for_status()
    data = response.json()

    # FIXED: handle both content-reply and tool-call-only replies
    message = data.get("choices", [{}])[0].get("message", {})
    return message.get("content") or ""


def ask_llm(query: str) -> str | None:
    # Rebuild system message with fresh scratchpad state for this turn
    fresh_system = {
        "role":    "system",
        "content": prompt_builder.build_prompt(query),   # CHANGED: pass query
    }
    if messages and messages[0]["role"] == "system":
        messages[0] = fresh_system
    else:
        messages.insert(0, fresh_system)

    messages.append({"role": "user", "content": query})

    try:

        # FIX 2: trim history before sending prevents unbounded context growth
        trimmed = _trim_history(messages)

        assistant_reply = request_completion(trimmed)

        # FIX 3: treat empty response as a recoverable situation, not an exception.
        # Print a clear diagnostic instead of raising ValueError which was
        # caught by the generic except and printed as "Unexpected error".
        if not assistant_reply or not assistant_reply.strip():
            print(
                "[llm_client] Empty response from model.\n"
                "  Likely cause: context window overflow or model busy.\n"
                "  Input tokens were too close to the model's context limit.\n"
                "  Try a shorter message, or check that llama-server --ctx-size "
                "is at least 8192."
            )
            return (
                "I couldn't generate a response — my context window is probably too full. "
                "Try splitting your message into smaller parts."
            )

        print(assistant_reply)
        tool_executor.execute_tool_calls(assistant_reply)
        parsed_response = parse_response(assistant_reply)

        if not parsed_response.strip():
            # Retry without tools — model replied with only tool calls, no text
            retry_messages = [
                {"role": "system", "content": prompt_builder.build_prompt(query)},
                {"role": "user", "content": query},
                {"role": "assistant", "content": assistant_reply},
                {"role": "user", "content": "Now give me a normal text reply without any tool calls."},
            ]
            retry_reply = request_completion(retry_messages)
            if retry_reply:
                parsed_response = parse_response(retry_reply)

        if not parsed_response.strip():
            parsed_response = (
                "I could not produce a response without tool calls. "
                "Please rephrase your request."
            )

        messages.append({"role": "assistant", "content": assistant_reply})

        # FIX 4: memory extraction was running on EVERY turn using the same
        # model port, adding ~1000 tokens of extra load immediately after the
        # main call. If the main call already stressed the context, this second
        # call would also fail or slow down the loop significantly.
        # Now runs only when the reply contains real user-relevant content
        # (skip greetings, tool-only replies, and error messages).

        # Later change this to a seperate LLM to get relevant infromation, so that we can 
        # avoid excessive load and preserve the context window
        _maybe_extract_memory(query, assistant_reply)
 
        print(get_scratchpad_memory())
        return parsed_response

    except requests.exceptions.ConnectionError:
        print("[ERROR] Could not connect to local LLM server at :8081.")

    except requests.exceptions.Timeout:
        print("[ERROR] Model timed out. It may be overloaded or the context is too large.")

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Request failed: {e}")

    except Exception as e:
        print(f"[ERROR] Unexpected error: {type(e).__name__}: {e}")

def _maybe_extract_memory(user_message: str, assistant_reply: str) -> None:
    """
    FIX 4: Only run memory extraction when there's a real chance of
    extractable facts. Skip short/greeting turns to reduce model load.
    Extraction failure is always non-fatal.
    """
    # Skip if the user message is too short to contain extractable facts
    if len(user_message.split()) < 8:
        return
 
    # Skip if assistant reply is mostly tool calls with no real content
    stripped = assistant_reply
    import re
    stripped = re.sub(r"<tool_call>.*?</tool_call>", "", stripped, flags=re.DOTALL).strip()
    if len(stripped.split()) < 5:
        return
 
    try:
        extract_and_store(
            user_message=user_message,
            assistant_reply=assistant_reply,
        )
    except Exception as e:
        print(f"[llm_client] Memory extraction error (non-fatal): {e}")


if __name__ == "__main__":
    while True:
        user_query = input("You: ")

        if user_query.strip().lower() == "/stop":
            process_manager.stop_from_cli()
            break

        answer = ask_llm(user_query)

        print("\nAssistant:")
        print(answer)
