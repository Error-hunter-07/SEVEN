"""
CHANGED: build_dynamic_context() (formerly build_prompt()) passes
`user_query` down to memory_retriever.get_retrieved_context() so semantic
memory can do a similarity search against the current user message.

CHANGED (episodic memory): build_dynamic_context() also runs the
deterministic episodic-recall trigger (LLMEngine/episodic_trigger.py)
against the raw user_query BEFORE the LLM ever sees it. If the query
matches recall-shaped phrasing ("what did we discuss", "last time",
etc.), compiled episodic context gets appended to retrieved_context
automatically — the LLM doesn't have to notice it needs to call
search_episodic_memory for the obvious cases. Anything outside those
patterns still relies on the LLM choosing to call the tool itself.

CHANGED (prompt caching): this module no longer combines SYSTEM_PROMPT
with the retrieved context into one string. SYSTEM_PROMPT is exported as
a plain constant and set as the system message ONCE per session by
LLMEngine.llm_client.ask_llm; build_dynamic_context() returns only the
per-turn retrieved/episodic context, which the caller now attaches to
the current user turn instead. This keeps the system message — and the
unchanged tail of prior conversation history — byte-identical across
turns so llama-server's --cache-prompt/--cache-reuse can actually reuse
the KV cache for them instead of reprocessing the whole prompt on every
turn. See build_dynamic_context()'s own docstring for the full reasoning.
"""

import MemoryManagement.memory_retriever as memory_retriever
import GlobalHelpers.token_counter as token_counter
import LLMEngine.episodic_trigger as episodic_trigger
from GlobalHelpers.logger import get_logger

log = get_logger(__name__)

LOCAL_CTX_LIMIT = 12000

SYSTEM_PROMPT = """
You are Seven (Female), an advance AI assistant similar to Jarvis. You are designed to help the user with a wide range of tasks from answering easy questions to totally
discovering new knowledge. You are highly skilled and you are very concise with you answers. You are very creative and you are very good at coming up with new ideas.
You have many tools to help you with your tasks, it is important that you utilize them to the best of your ability. 
Make sure to use your tools whevever possible. The tools will help you to retain your memory and remember things about the user.
 
TOOL CALL FORMAT:
<tool_call>
{"tool": "tool_name", "arguments": {"key": "value"}}
</tool_call>
 
MEMORY TYPES — you have three different kinds of memory, use the right one:
 
- WORKING MEMORY (add_scratchpad_memory_update, update=False): structured,
  session-scoped facts you'll likely need to reference again LATER IN THIS
  SAME SESSION — a name, a location, an age, an ongoing project's details,
  a budget number, a decision already made. Think of it as short-term
  working notes for the task at hand. Use it liberally whenever the user
  states a concrete fact you might need to recall in a few turns.
 
- SEMANTIC MEMORY (store_semantic_memory / search_semantic_memory): durable
  SINGLE FACTS about the user that should persist ACROSS SESSIONS, even
  after this conversation ends — identity, long-term preferences, skills,
  goals, relationships. Use search_semantic_memory when you need ONE
  standing fact (e.g. "what's my budget", "what do I do for work").

- EPISODIC MEMORY (search_episodic_memory / browse_episodic_memory): what happened in a PAST
  CONVERSATION — a process, a decision, a sequence of events, alternatives
  that were considered and why one was chosen. Use search_episodic_memory
  for topic-based recall ("what design were we considering"). Use
  browse_episodic_memory when the user cares about WHEN or WHICH session,
  not what topic — recent history, the oldest sessions, a specific past
  session by id, or a time window (e.g. "last 7 days"). Semantic memory
  cannot answer these — it only stores standing facts, not the reasoning,
  sequence, or timing behind them.
 
These are not mutually exclusive — many facts belong in more than one.
When in doubt, store to working memory (cheap, session-scoped) even if
you're unsure it also deserves semantic memory.
 
WHEN TO USE TOOLS:
- Any task/request → update_scratchpad_state planning.current_goal
- Multi-step task → update_scratchpad_state planning.subtasks
- Multi-step task in progress → periodically check in with the user ("Should I mark [subtask] as done?") rather than silently guessing — only call update_scratchpad_state planning.completed_subtasks once the user confirms
- User states any concrete fact you may need again this session (name, numbers, location, decisions, constraints) → add_scratchpad_memory_update (update=False)
- User shares personal info (name, skills, project, preferences) → add_scratchpad_memory_update AND store_semantic_memory
- Long-term fact about user → store_semantic_memory (importance 0.4-1.0)
- User asks for a personalized suggestion/recommendation ("you know me", "what would I like", "pick for me") → search_semantic_memory BEFORE answering, don't guess
- User wants a SINGLE STANDING FACT recalled → search_semantic_memory
- User wants to recall a PROCESS, DECISION, or SEQUENCE from a past conversation, by TOPIC → search_episodic_memory
- User wants recent history, the earliest sessions, a specific past session, or a time-bounded slice ("what have we discussed this week", "the very first thing we talked about", "everything from that session") → browse_episodic_memory (mode: recent/oldest/semantic/by_session, plus within_days)
- Unsure which episodic tool fits → prefer search_episodic_memory for "what/why" topic questions, browse_episodic_memory for "when/which session" access-pattern questions
- Need to recall something set earlier this session → get_working_memory or get_all_working_memory
- Task complete / topic done → update_scratchpad_summary
- Error or failure → update_scratchpad_state execution.last_error
- Never answer in only tool calls, you must provide a natural language response as well.
 
STYLE: concise, calm, no filler phrases like "Certainly!" or "Great question!".
"""



def build_dynamic_context(user_query: str) -> str:
    """
    Returns ONLY the per-turn dynamic context (retrieved semantic memory +
    any triggered episodic recall) — NOT the static SYSTEM_PROMPT.

    CHANGED: previously named build_prompt() and returned
    f"{SYSTEM_PROMPT}\\n\\n{retrieved_context}", with the caller putting
    that whole combined string into the system message on every turn.
    That meant the system message (message index 0 -- the very start of
    the prompt) changed on almost every request, since retrieved_context
    varies per query. llama-server is started with --cache-prompt
    --cache-reuse (see Runtime/process_manager.py), which reuses the KV
    cache for unchanged prompt content -- but a prompt whose FIRST message
    changes every turn gets little benefit from that, since the most
    expensive part of the prompt (SYSTEM_PROMPT + retrieved context) was
    never a cache hit.

    Now the caller (LLMEngine.llm_client.ask_llm) keeps the system
    message fixed to the literal SYSTEM_PROMPT constant for the whole
    session, and glues this function's return value onto the CURRENT
    user turn's content instead (context immediately followed by the
    actual query, in the last message of the array). Only that small,
    always-new tail needs reprocessing each turn; the system message and
    the earlier, unchanged conversation history stay cache-eligible.

    The token-budget math below is unchanged from before -- it still
    accounts for SYSTEM_PROMPT's size even though this function no
    longer returns it, since SYSTEM_PROMPT still occupies real context
    window space in the actual request regardless of which message it
    lives in.
    """
    retrieved_context = memory_retriever.get_retrieved_context(query=user_query)

    # Deterministic episodic recall trigger — see module docstring.
    # Returns None on almost every turn (only fires on recall-shaped
    # phrasing), so this is a no-op most of the time.
    try:
        episodic_context = episodic_trigger.maybe_get_episodic_context(user_query)
    except Exception:
        log.exception("Episodic trigger raised unexpectedly (non-fatal, continuing without it).")
        episodic_context = None

    if episodic_context:
        retrieved_context = "\n\n".join(filter(None, [retrieved_context, episodic_context]))

    total = token_counter.count_tokens(
        SYSTEM_PROMPT + retrieved_context + user_query
    )

    if total > LOCAL_CTX_LIMIT:
        # Drop everything gathered above (semantic retrieval AND any
        # injected episodic context) and fall back to the cheapest
        # possible context — a query="" semantic retrieval — rather than
        # trying to selectively trim, which risks keeping a large
        # episodic block over a small, more relevant semantic snippet.
        retrieved_context = memory_retriever.get_retrieved_context(query="")
        total = token_counter.count_tokens(
            SYSTEM_PROMPT + retrieved_context + user_query
        )
        log.warning("Over limit (%d tokens) — dropped semantic memory and episodic context.", total)

    if total > LOCAL_CTX_LIMIT:
        retrieved_context = ""
        log.warning("Still over limit — dropped all context.")

    return retrieved_context