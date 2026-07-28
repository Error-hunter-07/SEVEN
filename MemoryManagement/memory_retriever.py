"""
Compiles all memory sources into a single context string for the system prompt.

CHANGED: now accepts the current user query so semantic memory can do
similarity search. prompt_builder.build_dynamic_context() passes the query here.

Context assembly order (matters for prompt position):
  1. Short-term (scratchpad) — always included, highest priority
  2. Long-term semantic       — injected when query is provided
"""

from MemoryManagement.shortterm_memory import scratchpad

# FIX 2: import the singleton, don't create a new instance
from MemoryManagement.semantic_memory.semantic_memory import semantic_memory


def get_retrieved_context(query: str = "") -> str:
    """
    Build the full memory context string for the current turn.

    Args:
        query: The current user message. Used to do semantic similarity
               search against long-term memory. Pass "" to skip semantic retrieval.

    Returns:
        A formatted string to inject into the system prompt.
    """
    parts = []

    # 1. Short-term scratchpad (always included)
    scratchpad_context = scratchpad.get_compiled_memory()
    if scratchpad_context and scratchpad_context.strip():
        parts.append(scratchpad_context)

    # 2. Long-term semantic memory (only when we have a query to search with)
    if query and query.strip():
        semantic_context = semantic_memory.retrieve_as_text(
            query=query,
            k=5,
            min_importance=0.4,   # skip low-value memories to save tokens
        )
        if semantic_context:
            parts.append(semantic_context)

    return "\n\n".join(parts)