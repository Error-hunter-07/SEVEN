"""
CHANGED: build_prompt() now passes `user_query` down to
memory_retriever.get_retrieved_context() so semantic memory can do a
similarity search against the current user message.

Previously it passed "" always (query was unused).
"""

import MemoryManagement.memory_retriever as memory_retriever
import GlobalHelpers.token_counter as token_counter

SYSTEM_PROMPT = SYSTEM_PROMPT = """
You are Seven (Female), an advance AI assistant similar to Jarvis. You are designed to help the user with a wide range of tasks from answering easy questions to totally
discovering new knowledge. You are highly skilled and you are very concise with you answers. You are very creative and you are very good at coming up with new ideas.
You have many tools to help you with your tasks, it is important that you utilize them to the best of your ability. 
Make sure to use your tools whevever possible. The tools will help you to retain your memory and remember things about the user.

TOOL CALL FORMAT:
<tool_call>
{"tool": "tool_name", "arguments": {"key": "value"}}
</tool_call>
"""


def build_prompt(user_query: str) -> str:
    # CHANGED: pass user_query so semantic memory can retrieve relevant facts
    retrieved_context = memory_retriever.get_retrieved_context(query=user_query)

    total_tokens = token_counter.count_tokens(SYSTEM_PROMPT + retrieved_context + user_query)

    if total_tokens > 125000:
        # CHANGED: instead of wiping all context, try dropping only semantic memory
        # by re-fetching with no query (scratchpad only)
        retrieved_context = memory_retriever.get_retrieved_context(query="")
        print("[PromptBuilder] Token limit hit — dropped semantic memory from context.")

        if token_counter.count_tokens(SYSTEM_PROMPT + retrieved_context + user_query) > 125000:
            retrieved_context = ""
            print("[PromptBuilder] Token limit hit — dropped all context.")

    if retrieved_context:
        return f"{SYSTEM_PROMPT}\n\n{retrieved_context}"
    return SYSTEM_PROMPT
