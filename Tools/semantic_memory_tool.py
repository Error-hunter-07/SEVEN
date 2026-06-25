"""
Bridge between the LLM and SemanticMemory — mirrors working_memory_tool.py.

The LLM never touches ChromaDB directly.
It calls these functions through tool_calls, exactly like it does for
working memory (add_scratchpad_memory_update_flat).

Two tools are exposed to the LLM:
  1. store_semantic_memory  — save an important fact
  2. search_semantic_memory — search past memories by topic
"""

from MemoryManagement.semantic_memory.semantic_memory import SemanticMemory
import Tools.scratchpad_tool as scratchpad_tool

_semantic_memory = SemanticMemory()


def store_semantic_memory(
    text: str,
    importance: float = 0.5,
    category: str     = "other",
) -> str | None:
    """
    Store a distilled long-term memory fact.

    Args:
        text:       A self-contained fact sentence about the user.
        importance: 0.0–1.0. Use 0.8+ for identity/goals, 0.5 for general facts.
        category:   One of: identity, education, interests, goals,
                    preferences, experience, relationships, other.

    Returns:
        The new memory id on success, None on failure.
    """
    mem_id = _semantic_memory.store(
        text=text,
        importance=importance,
        category=category,
        source="llm_tool_call",
    )
    scratchpad_tool.add_scratchpad_tool_output(
        "store_semantic_memory",
        f"Stored semantic memory id={mem_id}: {text[:80]}" if mem_id
        else "Failed to store semantic memory.",
    )
    return mem_id


def search_semantic_memory(query: str, k: int = 5) -> str:
    """
    Search long-term semantic memory for facts relevant to `query`.
    Returns a formatted string of results (or a 'no results' message).

    The LLM can call this before answering questions about the user's
    background, preferences, or past experiences.
    """
    results = _semantic_memory.retrieve(query=query, k=k)

    if not results:
        output = "No relevant long-term memories found."
    else:
        lines = []
        for i, r in enumerate(results, 1):
            meta = r["metadata"]
            lines.append(
                f"{i}. [{meta.get('category', '')}] {r['text']} "
                f"(importance: {meta.get('importance', '')})"
            )
        output = "\n".join(lines)

    scratchpad_tool.add_scratchpad_tool_output("search_semantic_memory", output)
    return output
