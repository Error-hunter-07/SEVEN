"""
Called at the END of every assistant turn (inside llm_client.ask_llm).
Takes the raw user message + assistant reply, calls the local LLM to extract
1-3 distilled facts, then stores them via SemanticMemory.

Extraction is a SEPARATE small LLM call — not part of the main conversation.
We NEVER embed raw chat logs.  We embed distilled fact sentences only.
"""

from __future__ import annotations

import json
import requests

from MemoryManagement.semantic_memory.semantic_memory import semantic_memory  # FIX 2: singleton
from GlobalHelpers.logger import get_logger
from GlobalHelpers.config import settings

log = get_logger(__name__)

# ── Extraction prompt ────────────────────────────────────────────────────────

_EXTRACTION_SYSTEM = """You are a memory extraction assistant.
Given a conversation snippet, extract 1 to 3 important, self-contained facts about the USER that are worth remembering long-term.

Rules:
- Write each fact as a single, complete sentence about the user.
- Only extract facts that would still be useful months later.
- Skip greetings, filler, one-off commands, and tool call outputs.
- Do NOT extract facts about yourself (Seven) or general knowledge.
- Assign an importance score 0.0–1.0 (1.0 = critical identity/preference, 0.5 = moderately useful).
- Assign a category: one of: identity, education, interests, goals, preferences, experience, relationships, other.
- Assign a polarity: "positive" (user likes/wants/prefers), "negative" (user dislikes/opposes/avoids), or "neutral" (factual, no preference signal).

Respond ONLY with a valid JSON array. No explanation, no markdown fences.

Example output:
[
  {"memory": "User dislikes Python.", "importance": 0.6, "category": "preferences", "polarity": "negative"},
  {"memory": "User is learning Java and C++.", "importance": 0.70, "category": "interests", "polarity": "neutral"}
]

If there is nothing worth remembering, respond with an empty array: []
"""

def extract_and_store_batch(turns: list[tuple[str, str]]) -> None:
    """
    Same as extract_and_store, but takes a list of (user_msg, assistant_reply)
    turns accumulated during a cooldown window and extracts facts from all of
    them in ONE LLM call instead of dropping them.
    """
    turns = [t for t in turns if t[0] or t[1]]
    if not turns:
        return

    # Concatenate turns into one conversation snippet, in order
    conversation_snippet = "\n".join(
        f"User: {u}\nAssistant: {_strip_tool_calls(a)}"
        for u, a in turns
    )

    # Guard against unbounded prompt growth 
    MAX_SNIPPET_CHARS = 6000
    if len(conversation_snippet) > MAX_SNIPPET_CHARS:
        conversation_snippet = conversation_snippet[-MAX_SNIPPET_CHARS:]

    facts = _extract_facts_from_snippet(conversation_snippet)
    for fact in facts:
        memory_text = fact.get("memory", "").strip()
        if not memory_text:
            continue
        semantic_memory.store(
            text=memory_text,
            importance=float(fact.get("importance", 0.5)),
            category=fact.get("category", "other"),
            polarity=fact.get("polarity", "neutral"),
            source="conversation",
        )

# Refactor _extract_facts to take a pre-built snippet, so both
# extract_and_store and extract_and_store_batch can share it:
def _extract_facts_from_snippet(conversation_snippet: str) -> list[dict]:
    try:
        response = requests.post(
            "http://127.0.0.1:8081/v1/chat/completions",
            json={
                "model": settings.llm_model,
                "messages": [
                    {"role": "system", "content": _EXTRACTION_SYSTEM},
                    {"role": "user", "content": conversation_snippet},
                ],
                "temperature": 0.2,
                "max_tokens": 512,
            },
            timeout=30,
        )
        response.raise_for_status()
        raw_text = response.json().get("choices", [{}])[0].get("message", {}).get("content", "[]")
        return _parse_json_array(raw_text)
    except requests.exceptions.RequestException as e:
        log.error("LLM call failed: %s", e, exc_info=True)
        return []
    except Exception as e:
        log.error("Unexpected error: %s", e, exc_info=True)
        return []

def extract_and_store(user_message: str, assistant_reply: str) -> None:
    """
    Main entry point. Call this after every assistant turn.

    Args:
        user_message:    The raw user query for this turn.
        assistant_reply: The raw assistant response (including tool call tags).
    """
    if not user_message and not assistant_reply:
        return

    facts = _extract_facts(user_message, assistant_reply)
    if not facts:
        return

    for fact in facts:
        memory_text = fact.get("memory", "").strip()
        importance  = float(fact.get("importance", 0.5))
        category    = fact.get("category", "other")
        polarity    = fact.get("polarity", "neutral")

        if not memory_text:
            continue

        semantic_memory.store(
            text=memory_text,
            importance=importance,
            category=category,
            polarity=polarity,
            source="conversation",
        )


def _extract_facts(user_message: str, assistant_reply: str) -> list[dict]:
    """
    Calls the local LLM with a focused extraction prompt.
    Returns a list of {memory, importance, category, polarity} dicts.
    """
    conversation_snippet = (
        f"User: {user_message}\n"
        f"Assistant: {_strip_tool_calls(assistant_reply)}"
    )

    try:
        response = requests.post(
            "http://127.0.0.1:8081/v1/chat/completions",
            json={
                "model":       settings.llm_model,
                "messages": [
                    {"role": "system", "content": _EXTRACTION_SYSTEM},
                    {"role": "user",   "content": conversation_snippet},
                ],
                "temperature": 0.2,   # low temp for precise facts
                "max_tokens":  512,
            },
            timeout=30,
        )
        response.raise_for_status()
        raw_text = (
            response.json()
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content", "[]")
        )
        return _parse_json_array(raw_text)

    except requests.exceptions.RequestException as e:
        log.error("LLM call failed: %s", e, exc_info=True)
        return []
    except Exception:
        log.exception("Unexpected error in extract_and_store")
        return []


def _strip_tool_calls(text: str) -> str:
    """Removing <tool_call>...</tool_call> blocks before sending to extractor."""
    import re
    return re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL).strip()


def _parse_json_array(text: str) -> list[dict]:
    """Parsing the LLM response as a JSON array. Returns [] on any failure."""
    text = text.strip()
    # Strip accidental markdown fences
    if text.startswith("```"):
        text = "\n".join(
            line for line in text.splitlines()
            if not line.strip().startswith("```")
        ).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        return []
    except json.JSONDecodeError as e:
        log.warning("JSON parse failed: %s — raw: %s", e, text[:200])
        return []
