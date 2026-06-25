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


def _build_system_message() -> dict:
    """Build the system message dict with a freshly compiled prompt."""
    return {
        "role":    "system",
        "content": prompt_builder.build_prompt("")
    }


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
            "max_tokens":  2048,
        },
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
        assistant_reply = request_completion(messages)

        if not assistant_reply:
            raise ValueError("Empty response from model")

        print(assistant_reply)
        tool_executor.execute_tool_calls(assistant_reply)
        parsed_response = parse_response(assistant_reply)

        if not parsed_response.strip():
            retry_prompt = prompt_builder.build_prompt(query)
            retry_messages = [
                {"role": "system",  "content": retry_prompt},
                {"role": "user",    "content": query},
                {
                    "role": "system",
                    "content": (
                        "Return a normal text answer. "
                        "Do not include any <tool_call> blocks."
                    )
                },
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

        # CHANGED: extract and store long-term memories from this turn.
        # Runs after the reply is committed so it never delays the response.
        # Passes the raw assistant_reply (tool calls stripped inside extractor).
        try:
            extract_and_store(
                user_message=query,
                assistant_reply=assistant_reply,
            )
        except Exception as e:
            # Never let extraction failure break the main conversation loop
            print(f"[llm_client] Memory extraction error (non-fatal): {e}")

        print(get_scratchpad_memory())
        return parsed_response

    except requests.exceptions.ConnectionError:
        print("[ERROR] Could not connect to local LLM.")

    except requests.exceptions.Timeout:
        print("[ERROR] Model took too long to respond.")

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Request failed: {e}")

    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")


if __name__ == "__main__":
    while True:
        user_query = input("You: ")

        if user_query.strip().lower() == "/stop":
            process_manager.stop_from_cli()
            break

        answer = ask_llm(user_query)

        print("\nAssistant:")
        print(answer)
