"""
LLMEngine/history_manager.py

Owns the conversation history list and the trimming logic that keeps it
inside the model's context window. Previously this state and logic lived
directly inside llm_client.py, mixed in with process bootstrap, HTTP
request building, and the background extraction queue — this module
gives it one focused job.
"""

# Keep only the last N user/assistant pairs. System message is always kept.
MAX_HISTORY_TURNS = 4

messages: list[dict] = []


def set_system_message(content: str) -> None:
    """Replaces the existing system message (position 0), or inserts one
    if none exists yet."""
    fresh_system = {"role": "system", "content": content}
    if messages and messages[0]["role"] == "system":
        messages[0] = fresh_system
    else:
        messages.insert(0, fresh_system)


def append_user_message(content: str) -> None:
    messages.append({"role": "user", "content": content})


def append_message(message: dict) -> None:
    """For appending pre-built messages, e.g. the assistant's history entry
    (which may include tool_calls)."""
    messages.append(message)


def get_full_history() -> list[dict]:
    return messages


def get_trimmed_history() -> list[dict]:
    """
    Keep system message + last MAX_HISTORY_TURNS of user/assistant pairs.
    Never mutates the original list.
    """
    system = [m for m in messages if m["role"] == "system"]
    conversation = [m for m in messages if m["role"] != "system"]
    trimmed = conversation[-(MAX_HISTORY_TURNS * 2):]
    return system + trimmed
