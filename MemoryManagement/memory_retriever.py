from .shortterm_memory import conversation_history as conversation
from .shortterm_memory import scratchpad

conversation_summary = conversation.conversation_summary
scratchpad_memory = scratchpad.scratchpad_memory

def get_retrieved_context():
    return conversation.conversation_summary + "\n\n" + scratchpad_memory

    