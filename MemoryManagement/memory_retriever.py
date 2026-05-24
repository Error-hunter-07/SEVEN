from .shortterm_memory import conversation_history as conversation
from .shortterm_memory import scratchpad

def get_retrieved_context():
    return conversation.conversation_summary + "\n\n" + scratchpad.get_compiled_memory()

    