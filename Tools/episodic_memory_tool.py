"""
Tools/episodic_memory_tool.py

Bridge between the LLM and episodic memory — mirrors
semantic_memory_tool.py's and working_memory_tool.py's shape.

One tool exposed to the LLM:
  search_episodic_memory — semantic search over past session summaries,
                            with each matched episode's linked semantic
                            facts (related_semantic_memory_ids) compiled
                            in alongside it.

This is the FULL RECALL tool — for a single standing fact, the LLM
should use search_semantic_memory instead; this tool is for recalling a
PROCESS, DECISION, or SEQUENCE from a specific past conversation (e.g.
"what design were we considering for Rema's tutorial website" — the
kind of question semantic memory's atomized facts structurally can't
answer, since it's about reasoning/sequence, not a standing fact).

search_and_compile_episodic_context() is the shared logic, called from
BOTH this tool AND LLMEngine/episodic_trigger.py's deterministic
pattern-match trigger — one function, two entry points, so they can
never silently drift apart in behavior.
"""

import MemoryManagement.episodic_memory.episodic_memory_store as episodic_memory_store
from MemoryManagement.semantic_memory.semantic_memory import semantic_memory
import Tools.scratchpad_tool as scratchpad_tool
from GlobalHelpers.logger import get_logger

log = get_logger(__name__)

MAX_FACTS_PER_EPISODE = 5


def _compile_episode_with_facts(episode: dict) -> str:
    """Renders one matched episode plus up to MAX_FACTS_PER_EPISODE of
    its linked semantic-memory facts into one readable block."""
    lines = [f"Episode: {episode.get('title') or '(untitled)'}"]
    if episode.get("start_time_iso"):
        lines.append(f"  When: {episode['start_time_iso'][:10]}")
    lines.append(f"  Summary: {episode.get('summary') or ''}")

    related_ids = (episode.get("related_semantic_memory_ids") or [])[:MAX_FACTS_PER_EPISODE]
    if related_ids:
        try:
            facts = semantic_memory.get_by_ids(related_ids)
        except Exception:
            log.exception("Failed to resolve related_semantic_memory_ids for episode %s (non-fatal).", episode.get("id"))
            facts = []
        if facts:
            lines.append("  Related facts:")
            for f in facts:
                lines.append(f"    - {f['text']}")

    return "\n".join(lines)


def search_and_compile_episodic_context(query: str, k: int = 3) -> str:
    """
    Shared search-and-compile logic. Semantically searches episode
    title+summary text, and for each match, pulls its linked semantic
    facts and compiles everything into one readable block.

    Bumps each matched episode's access_count/last_accessed via
    mark_recalled() — deliberate, explicit-recall-only tracking. Never
    call this path from a passive seed; see episodic_memory_store's
    mark_recalled() docstring for why that discipline matters (same
    lesson as working_memory's dead access_count column).

    Non-fatal throughout: any failure returns "" rather than raising, so
    a broken search never blocks the turn it was trying to help.
    """
    if not query or not query.strip():
        return ""

    try:
        episodes = episodic_memory_store.search_episodes(query, k=k)
    except Exception:
        log.exception("search_episodes failed for query '%s' (non-fatal).", query[:80])
        return ""

    if not episodes:
        return ""

    blocks = []
    for ep in episodes:
        blocks.append(_compile_episode_with_facts(ep))
        try:
            episodic_memory_store.mark_recalled(ep["id"])
        except Exception:
            log.exception("mark_recalled failed for episode %s (non-fatal).", ep.get("id"))

    return "\n\n".join(blocks)


def search_episodic_memory(query: str, k: int = 3) -> str:
    """
    Search past session summaries for a topic, process, or decision —
    NOT a single standing fact (use search_semantic_memory for that).

    Use this when the user is asking to recall something that happened
    across a conversation: what was discussed, what was decided, what
    alternatives were considered, what design/approach was chosen and
    why. Each result includes the episode's summary plus any specific
    facts tied to that session.

    Args:
        query: What to search for, e.g. "website design for Rema's tutorial".
        k:     Max number of episodes to return (default 3).

    Returns:
        A formatted string of matched episodes with their linked facts,
        or a 'no results' message.
    """
    output = search_and_compile_episodic_context(query, k=k)
    if not output:
        output = "No relevant past sessions found."

    scratchpad_tool.add_scratchpad_tool_output("search_episodic_memory", output)
    return output