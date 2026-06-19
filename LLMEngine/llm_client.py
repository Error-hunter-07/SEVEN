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

load_dotenv()


# ---------------------------------------------------------------------------
# Conversation history
# CHANGED: `messages` now persists across turns instead of being cleared
# after every ask_llm() call. This is essential for KV-cache continuity:
# llama.cpp's --cache-reuse works by matching the longest common prefix of
# tokens between successive requests. If we clear the list each turn the
# prefix is always just [system], so only the system prompt tokens are
# reused. With history retained the prefix grows turn-by-turn and the cache
# reuses all prior turns — exactly what --cache-prompt + --cache-reuse 256
# is designed for.
#
# The system message lives at index 0 and is REPLACED (not re-inserted) at
# the start of each turn with a freshly built prompt so the scratchpad state
# is always current. This keeps the system message token position stable,
# which is critical for cache hits — shifting token positions invalidates the
# cache for all tokens after the shift point.
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
        "role": "system",
        "content": prompt_builder.build_prompt("")
    }

def _build_tool_schema(parameters: dict) -> dict:
    """Convert the flat {param_name: description} dict into a valid JSON Schema."""
    if not parameters:
        return {"type": "object", "properties": {}}
    
    properties = {}
    for name, desc in parameters.items():
        # Infer type hint from the description prefix
        desc_str = str(desc)
        if desc_str.startswith("bool"):
            prop_type = "boolean"
        elif desc_str.startswith("int"):
            prop_type = "integer"
        else:
            prop_type = "string"
        properties[name] = {"type": prop_type, "description": desc_str}
    
    return {
        "type": "object",
        "properties": properties,
        "required": list(parameters.keys())
    }


def request_completion(request_messages):
    response = requests.post(
        "http://127.0.0.1:8081/v1/chat/completions",
        json={
            "model": os.getenv("LLM_MODEL"),
            "messages": request_messages,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": _build_tool_schema(tool.parameters)
                    }
                }
                for tool in registry.list_tools()
            ],
            "temperature": 0.7,
            "max_tokens": 2048
        },
    )
    response.raise_for_status()
    data = response.json()
    return (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content")
    )


def ask_llm(query):
    # CHANGED: System message is REPLACED at index 0 on every turn (not
    # inserted). This keeps the scratchpad state current in the prompt while
    # leaving all prior user/assistant turns in place for KV cache reuse.
    # Old code did messages.insert(0, ...) which would accumulate duplicate
    # system messages if the list weren't cleared — now clearing is no longer
    # needed and the system slot is always exactly one entry at position 0.
    fresh_system = _build_system_message()
    if messages and messages[0]["role"] == "system":
        # Replace the existing system message with the freshly built one.
        messages[0] = fresh_system
    else:
        # First turn: no system message yet — prepend it.
        messages.insert(0, fresh_system)

    # Add the new user turn.
    messages.append({
        "role": "user",
        "content": query
    })

    try:
        assistant_reply = request_completion(messages)
        if assistant_reply is None:
            raise KeyError("No message content in response")

        print(assistant_reply)
        tool_executor.execute_tool_calls(assistant_reply)
        parsed_response = parse_response(assistant_reply)

        if not parsed_response.strip():
            retry_prompt = prompt_builder.build_prompt(query)
            retry_messages = [
                {"role": "system", "content": retry_prompt},
                {"role": "user", "content": query},
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

        # CHANGED: Append the assistant reply to history so the next turn
        # includes it in the message list. This is what allows KV cache reuse
        # across turns — the model sees the full prior context each request
        # and llama.cpp reuses all matching prefix tokens from the cache.
        messages.append({
            "role": "assistant",
            "content": assistant_reply
        })

        # CHANGED: Removed messages.clear() — clearing was the root cause of
        # SEVEN having no conversational memory. History is now preserved.
        # The system message at index 0 is updated each turn (see above) so
        # there is no need to wipe the list.

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
