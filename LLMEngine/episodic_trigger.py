"""
LLMEngine/episodic_trigger.py

Deterministic trigger layer for episodic memory recall — runs BEFORE
the LLM ever sees the user's message, so obvious recall-shaped phrasing
("what did we discuss", "last time", "continue from where we left off")
gets episodic context injected automatically, without depending on the
LLM choosing to call search_episodic_memory itself. This is the safety
net for exactly the failure mode seen earlier in this project (the "you
know me well, suggest a bike color" question that skipped a search it
should have made).

This is a safety net, not a replacement for the tool: phrasing outside
these patterns still relies on the LLM noticing it needs episodic
context and calling search_episodic_memory directly (see
Tools/episodic_memory_tool.py) — e.g. "what design were we considering
for Rema's tutorial website" doesn't match any of these patterns at
all, but should still trigger episodic search via the tool path. Trigger
+ tool together cover both the obvious-phrasing case and the
natural-question case.

Calls the SAME search_and_compile_episodic_context() function the tool
uses (Tools/episodic_memory_tool.py), so both entry points behave
identically rather than drifting apart over time.
"""

import re

import Tools.episodic_memory_tool as episodic_memory_tool
from GlobalHelpers.logger import get_logger

log = get_logger(__name__)

# Deliberately conservative. False negatives (missing a recall-shaped
# message) just mean the LLM has to reach for the tool itself, which
# still works — not a dead end. False positives (firing on a message
# that wasn't really asking to recall anything) waste one search call
# and a bit of context space, which is cheap. Tuned toward precision
# over exhaustive recall for that reason.
_TRIGGER_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\blast time\b",
        r"\bwe (discussed|talked about|decided|considered)\b",
        r"\bwhat did we\b",
        r"\bcontinue from where we left off\b",
        r"\bprevious(ly)? (session|conversation|discussion)\b",
        r"\bwhat have we (discussed|covered|talked about)\b",
        r"\bremind me what\b",
        r"\bearlier (session|conversation|we)\b",
        r"\bwhat was (the|our) (plan|design|decision|approach)\b",
    ]
]


def maybe_get_episodic_context(user_query: str, k: int = 2) -> str | None:
    """
    Returns compiled episodic context if user_query matches a recall
    trigger pattern, else None. Non-fatal on any failure — a broken
    trigger should never block a normal turn from proceeding; it just
    means this turn doesn't get the automatic context boost.
    """
    if not user_query or not user_query.strip():
        return None
    if not any(p.search(user_query) for p in _TRIGGER_PATTERNS):
        return None

    try:
        context = episodic_memory_tool.search_and_compile_episodic_context(user_query, k=k)
        if context:
            log.debug("Deterministic episodic trigger fired for query: %s", user_query[:80])
        return context or None
    except Exception:
        log.exception("Episodic trigger failed (non-fatal, continuing without injected context).")
        return None