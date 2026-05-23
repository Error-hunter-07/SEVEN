import requests
from MemoryManagement.shortterm_memory.conversation_history import trim_context
from .response_parser import parse_response
import os
from dotenv import load_dotenv
import PrompBuilder.prompt_builder as prompt_builder

import ToolCalling.executor as tool_executor

messages = []

load_dotenv()  # Load environment variables from .env file

# Context window

def ask_llm(query):
    prompt = prompt_builder.build_prompt(query)
    # Add user message
    messages.append({
        "role": "user",
        "content": query
    })
    # Trim context if needed
    trim_context(messages)

    temp_messages = messages.copy()
    temp_messages.insert(
        0,
        {
            "role": "system",
            "content": prompt
        }
    )

    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": os.getenv("LLM_MODEL"),
                "messages": temp_messages,
                "stream": False
            },
        
        )
        response.raise_for_status()
        data = response.json()
        assistant_reply = data["message"]["content"]
        messages.append({
            "role": "assistant",
            "content": assistant_reply
        })
        tool_executor.execute_tool_calls(assistant_reply)
        return parse_response(assistant_reply)  
    
    except ConnectionError:
        print("[ERROR] Could not connect to local LLM.")
    
    except TimeoutError:
        print("[ERROR] Model took too long to respond.")

    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")

    # Add assistant response

    # Trim again after assistant response
    trim_context(messages)



if __name__ == "__main__":
    while True:
        user_query = input("You: ")

        answer = ask_llm(user_query)

        print("\nAssistant:")
        print(answer)