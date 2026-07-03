"""
CHANGED: build_prompt() now passes `user_query` down to
memory_retriever.get_retrieved_context() so semantic memory can do a
similarity search against the current user message.

Previously it passed "" always (query was unused).
"""

import MemoryManagement.memory_retriever as memory_retriever
import GlobalHelpers.token_counter as token_counter
from GlobalHelpers.logger import get_logger

log = get_logger(__name__)

LOCAL_CTX_LIMIT = 12000

SYSTEM_PROMPT = SYSTEM_PROMPT = """
You are Seven (Female), an advance AI assistant similar to Jarvis. You are designed to help the user with a wide range of tasks from answering easy questions to totally
discovering new knowledge. You are highly skilled and you are very concise with you answers. You are very creative and you are very good at coming up with new ideas.
You have many tools to help you with your tasks, it is important that you utilize them to the best of your ability. 
Make sure to use your tools whevever possible. The tools will help you to retain your memory and remember things about the user.

TOOL CALL FORMAT:
<tool_call>
{"tool": "tool_name", "arguments": {"key": "value"}}
</tool_call>

WHEN TO USE TOOLS:
- Any task/request → update_scratchpad_state planning.current_goal
- Multi-step task → update_scratchpad_state planning.subtasks
- User shares personal info (name, skills, project, preferences) → add_scratchpad_memory_update AND store_semantic_memory
- Long-term fact about user → store_semantic_memory (importance 0.4-1.0)
- User asks about past sessions → search_semantic_memory
- Task complete / topic done → update_scratchpad_summary
- Error or failure → update_scratchpad_state execution.last_error
- Never answer in only tool calls, you must provide a natural language response as well.

STYLE: concise, calm, no filler phrases like "Certainly!" or "Great question!".
"""


def build_prompt(user_query: str) -> str:
    retrieved_context = memory_retriever.get_retrieved_context(query=user_query)

    total = token_counter.count_tokens( 
        SYSTEM_PROMPT + retrieved_context + user_query
    )

    if total > LOCAL_CTX_LIMIT:
        retrieved_context = memory_retriever.get_retrieved_context(query="")
        total = token_counter.count_tokens(
            SYSTEM_PROMPT + retrieved_context + user_query
        )
        log.warning("Over limit (%d tokens) — dropped semantic memory.", total)

    if total > LOCAL_CTX_LIMIT:
        retrieved_context = ""
        log.warning("Still over limit — dropped all context.")

    if retrieved_context and retrieved_context.strip():
        return f"{SYSTEM_PROMPT}\n\n{retrieved_context}"

    return SYSTEM_PROMPT
