import requests
from context_manager import trim_context
from response_parser import parse_response

SYSTEM_PROMPT = """ 
    Your Name is SEVEN.
   You are a personal expert AI Assistant for a BTech CSE student. You will serve like JARVIS from Iron Man.
    You will help the student with their studies, projects, and any other academic-related queries. You will provide detailed explanations, code snippets, and resources to assist the student in understanding complex concepts and completing their assignments effectively. Always be polite, patient, and encouraging in your responses.
    You will be his buddy and be cool to match his vibe. You will also help him with his personal life and be a good friend to him. You will always be there for him and support him in every way possible.
"""

# Context window
messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT
    }
]


def ask_llm(query):


    # Add user message
    messages.append({
        "role": "user",
        "content": query
    })

    # Trim context if needed
    trim_context(messages)


    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "gemma4:e4b",
            "messages": [
                {
                    "role":"system",
                    "content": SYSTEM_PROMPT
              },
                {
                    "role": "user",
                    "content": query
                }
            ],
            "stream": False
        }
    )
    data = response.json()
    assistant_reply = data["message"]["content"]

    # Add assistant response
    messages.append({
        "role": "assistant",
        "content": assistant_reply
    })

    # Trim again after assistant response
    trim_context(messages)


    return parse_response(assistant_reply)

if __name__ == "__main__":
    while True:
        user_query = input("You: ")

        answer = ask_llm(user_query)

        print("\nAssistant:")
        print(answer)