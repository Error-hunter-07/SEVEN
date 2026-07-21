"""
MemoryManagement/episodic_memory/summarizer.py

Generates the title/summary/key_topics for an episodic memory row via a
dedicated LLM call — same pattern as
MemoryManagement/semantic_memory/memory_extractor.py: a separate,
focused request to the local llama-server endpoint, not part of the
main conversation. This only ever fires once per session (at
on_session_end), once per crash-recovery sweep entry, or once per
decay-merge batch — never per-turn — so the extra LLM call is cheap
relative to how rarely it happens.

Three entry points, one per caller:
  summarize_session()  — normal clean session end
  summarize_crashed()  — crash-recovery sweep, working with whatever
                          durable working_memory rows survived
  summarize_merge()    — decay lifecycle, collapsing N old episodes
                          into one

All three return a dict: {"title": str, "summary": str, "key_topics": list[str]}
and are designed to NEVER return an unusable result — episodic_memory.summary
is NOT NULL, so every path falls back to a cheap heuristic string-join
(same style as the existing working-memory session summary in
session_lifecycle.py) if the LLM call fails for any reason. A failed
LLM call should degrade the quality of the summary, not block the
episode from being written at all.
"""

from __future__ import annotations

import json
import requests

from GlobalHelpers.logger import get_logger
from GlobalHelpers.config import settings

log = get_logger(__name__)

_LLM_ENDPOINT = "http://127.0.0.1:8081/v1/chat/completions"
_REQUEST_TIMEOUT = 30

_SESSION_SUMMARY_SYSTEM = """You are a memory summarization assistant.
Given a record of what happened during a conversation session, produce a short title, a 1-3 sentence summary, and a list of key topics.

Rules:
- title: max 8 words, no trailing punctuation.
- summary: 1-3 complete sentences, written about the USER's session (what they were doing, what got done, what broke).
- key_topics: 1-5 short lowercase topic strings (e.g. "ladakh trip", "budget planning").
- Do not invent details that aren't implied by the input.

Respond ONLY with a valid JSON object. No explanation, no markdown fences.
Example output:
{"title": "Ladakh trip budget planning", "summary": "User planned a trip to Ladakh and worked out a budget. Two subtasks were completed.", "key_topics": ["ladakh trip", "budget"]}
"""

_MERGE_SUMMARY_SYSTEM = """You are a memory summarization assistant.
Given several older episode summaries from past sessions, merge them into ONE combined title, a 2-4 sentence summary covering the recurring themes, and a list of key topics.

Rules:
- title: max 8 words, no trailing punctuation.
- summary: 2-4 complete sentences capturing the recurring themes across all the episodes, not a list of each one.
- key_topics: 1-6 short lowercase topic strings, deduplicated across the episodes.

Respond ONLY with a valid JSON object. No explanation, no markdown fences.
"""


def _call_llm(system_prompt: str, user_content: str) -> dict | None:
    try:
        response = requests.post(
            _LLM_ENDPOINT,
            json={
                "model": settings.llm_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0.2,
                "max_tokens": 400,
            },
            timeout=_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        raw_text = response.json().get("choices", [{}])[0].get("message", {}).get("content", "{}")
        return _parse_json_object(raw_text)
    except requests.exceptions.RequestException as e:
        log.error("Episodic summarizer LLM call failed: %s", e, exc_info=True)
        return None
    except Exception:
        log.exception("Unexpected error in episodic summarizer LLM call")
        return None


def _parse_json_object(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(
            line for line in text.splitlines() if not line.strip().startswith("```")
        ).strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError as e:
        log.warning("Episodic summarizer JSON parse failed: %s — raw: %s", e, text[:200])
        return None


def _normalize_result(result: dict | None, fallback: dict) -> dict:
    if not result:
        return fallback
    title = str(result.get("title") or fallback["title"]).strip()[:120]
    summary = str(result.get("summary") or fallback["summary"]).strip()
    if not summary:
        summary = fallback["summary"]
    key_topics = result.get("key_topics")
    if not isinstance(key_topics, list):
        key_topics = fallback["key_topics"]
    else:
        key_topics = [str(t).strip() for t in key_topics if str(t).strip()][:6]
    return {"title": title, "summary": summary, "key_topics": key_topics}


def summarize_session(goal, completed_subtasks, memory_updates, last_error, turn_count) -> dict:
    """Called from session_lifecycle.on_session_end for a clean shutdown."""
    parts = []
    if goal:
        parts.append(f"Goal: {goal}")
    if completed_subtasks:
        parts.append("Completed: " + "; ".join(str(t) for t in completed_subtasks[:5]))
    if memory_updates:
        parts.append("Notable updates: " + "; ".join(str(u) for u in memory_updates[:5]))
    if last_error:
        parts.append(f"Last error: {str(last_error)[:200]}")
    parts.append(f"Turn count: {turn_count}")

    fallback = {
        "title": (str(goal)[:60] if goal else "Session summary"),
        "summary": " | ".join(parts) if parts else "No notable activity this session.",
        "key_topics": [],
    }

    if not parts:
        return fallback

    result = _call_llm(_SESSION_SUMMARY_SYSTEM, "\n".join(parts))
    return _normalize_result(result, fallback)


def summarize_crashed(session_id, working_memory_snippets, turn_count) -> dict:
    """Called from session_lifecycle's crash-recovery sweep. Only durable
    breadcrumbs (working_memory rows already committed to SQLite) are
    available here — in-memory conversation history doesn't survive a
    crash."""
    fallback = {
        "title": "Interrupted session",
        "summary": (
            f"Session {session_id} ended without a clean shutdown after {turn_count} turn(s). "
            + (f"Last known state: {working_memory_snippets[:300]}" if working_memory_snippets else "No recoverable state.")
        ),
        "key_topics": [],
    }

    if not working_memory_snippets:
        return fallback

    user_content = f"Turn count: {turn_count}\nRecovered state: {working_memory_snippets[:2000]}"
    result = _call_llm(_SESSION_SUMMARY_SYSTEM, user_content)
    return _normalize_result(result, fallback)


def summarize_merge(episodes: list[dict]) -> dict:
    """Called from the decay lifecycle to collapse a batch of aged,
    same-level episodes into one summary row."""
    lines = []
    for ep in episodes:
        title = ep.get("title") or "(untitled)"
        summary = ep.get("summary") or ""
        lines.append(f"- {title}: {summary}")

    fallback = {
        "title": f"Merged history ({len(episodes)} episodes)",
        "summary": " ".join(str(ep.get("summary") or "") for ep in episodes)[:600] or "No summary available.",
        "key_topics": sorted({t for ep in episodes for t in (ep.get("key_topics") or [])})[:6],
    }

    if not lines:
        return fallback

    result = _call_llm(_MERGE_SUMMARY_SYSTEM, "\n".join(lines))
    return _normalize_result(result, fallback)
