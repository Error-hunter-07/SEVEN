import os
import sys

if __package__ is None or __package__ == "":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Tools.scratchpad_tool import get_scratchpad_memory
import requests
# from MemoryManagement.shortterm_memory.conversation_history import trim_context
try:
    from .response_parser import parse_response
except ImportError:
    from response_parser import parse_response
from dotenv import load_dotenv
import PrompBuilder.prompt_builder as prompt_builder
from Runtime.process_manager import ProcessManager

import ToolCalling.executor as tool_executor

messages = []
process_manager = None

load_dotenv()  # Load environment variables from .env file



process_manager = ProcessManager(
    model_path=os.getenv("LLM_MODEL_PATH"),
    llama_cli_path=os.getenv("LLM_CLI_PATH"),
    mmproj_path=os.getenv("MMPROJ_PATH")
)

try:
    process_manager.start_for_client()

except Exception as e:
    print(f"[ERROR] Failed to start LLM process: {e}")

def request_completion(request_messages):
    response = requests.post(
        "http://127.0.0.1:8081/v1/chat/completions",
        json={
            "model": os.getenv("LLM_MODEL"),
            "messages": request_messages,
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
    prompt = prompt_builder.build_prompt(query)
    # Add user message
    messages.append({
        "role": "user",
        "content": query
    })

    messages.insert(
        0,
        {
            "role": "system",
            "content": prompt
        }
    )

    try:
        assistant_reply = request_completion(messages)
        if assistant_reply is None:
            raise KeyError("No message content in response")

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

        print(get_scratchpad_memory())
        messages.clear()
        return parsed_response
    
    
    except requests.exceptions.ConnectionError:
        print("[ERROR] Could not connect to local LLM.")
    
    except requests.exceptions.Timeout:
        print("[ERROR] Model took too long to respond.")

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Request failed: {e}")

    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")

    # Add assistant response

    # Trim again after assistant response
    # trim_context(messages)



if __name__ == "__main__":
    while True:
        user_query = input("You: ")

        if user_query.strip().lower() == "/stop":
            process_manager.stop_from_cli()
            break

        answer = ask_llm(user_query)

        print("\nAssistant:")
        print(answer)