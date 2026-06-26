"""
Called at the END of every assistant turn (inside llm_client.ask_llm).
Takes the raw user message + assistant reply, calls the local LLM to extract
1-3 distilled facts, then stores them via SemanticMemory.

Extraction is a SEPARATE small LLM call — not part of the main conversation.
We NEVER embed raw chat logs.  We embed distilled fact sentences only.
"""

from __future__ import annotations

import json
import os
import requests

from MemoryManagement.semantic_memory.semantic_memory import semantic_memory  # FIX 2: singleton

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

Respond ONLY with a valid JSON array. No explanation, no markdown fences.

Example output:
[
  {"memory": "User is a first-year BTech CSE student.", "importance": 0.85, "category": "education"},
  {"memory": "User is learning Java and C++.", "importance": 0.70, "category": "interests"}
]

If there is nothing worth remembering, respond with an empty array: []
"""


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

        if not memory_text:
            continue

        semantic_memory.store(
            text=memory_text,
            importance=importance,
            category=category,
            source="conversation",
        )


def _extract_facts(user_message: str, assistant_reply: str) -> list[dict]:
    """
    Calls the local LLM with a focused extraction prompt.
    Returns a list of {memory, importance, category} dicts.
    """
    conversation_snippet = (
        f"User: {user_message}\n"
        f"Assistant: {_strip_tool_calls(assistant_reply)}"
    )

    try:
        response = requests.post(
            "http://127.0.0.1:8081/v1/chat/completions",
            json={
                "model":       os.getenv("LLM_MODEL"),
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
        print(f"[MemoryExtractor] LLM call failed: {e}")
        return []
    except Exception as e:
        print(f"[MemoryExtractor] Unexpected error: {e}")
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
        print(f"[MemoryExtractor] JSON parse failed: {e} — raw: {text[:200]}")
        return []
