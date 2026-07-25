"""
MemoryManagement/episodic_memory/summarizer.py

Generates episodic memory content via dedicated LLM calls — separate,
focused requests to the local llama-server, not part of the main
conversation. All calls route through LLMEngine.llm_request_lock, since
the local server runs --parallel 1 and can't process concurrent
requests; going through the shared lock keeps this summarizer's calls
from racing the main chat turn or the semantic-memory extractor.

Four entry points now, one per caller:
  summarize_chunk()    — rolling, every 5 turns, called by
                          LLMEngine/chunk_summary_worker.py. Short and
                          cheap (max_tokens=150) — a rough narrative
                          note, not a polished summary, since it gets
                          compressed again by summarize_session() at
                          session end.
  summarize_session()  — normal clean session end. Merges whatever
                          rolling chunk summaries exist (NOT the raw
                          transcript — by session end that's usually too
                          long to fit in one call, which is the whole
                          reason chunking happens live) plus scratchpad
                          signals (goal, completed subtasks, errors)
                          into one clean episode.
  summarize_crashed()  — crash-recovery sweep. Same inputs as
                          summarize_session() where available
                          (chunk_summaries), falling back to the raw
                          full_conversation backup if even the first
                          chunk never completed.
  summarize_merge()    — decay lifecycle, collapsing N old episodes
                          into one.

All four return {"title": str, "summary": str, "key_topics": list[str]}
and are designed to NEVER return an unusable result — every path falls
back to a cheap heuristic string-join if the LLM call fails, since a
failed LLM call should degrade summary quality, not block the episode
from being written at all.
"""

from __future__ import annotations

import json

from GlobalHelpers.logger import get_logger
from GlobalHelpers.config import settings
import LLMEngine.llm_request_lock as llm_request_lock

log = get_logger(__name__)

_REQUEST_TIMEOUT = 30

_CHUNK_SUMMARY_SYSTEM = """You are a memory summarization assistant.
Given a short slice of a conversation (a few turns), write ONE brief note (1-2 sentences) capturing what happened in THIS slice: what was discussed, decided, or done. This is an intermediate note, not a final summary — be concise, factual, and preserve any concrete decisions, options considered, or numbers mentioned.

Respond with ONLY the note text. No JSON, no markdown, no preamble."""

_SESSION_SUMMARY_SYSTEM = """You are a memory summarization assistant.
Given a sequence of notes describing what happened across a conversation session (in order), plus some structured signals, produce a title, a summary, and key topics.

Rules:
- title: max 8 words, no trailing punctuation.
- summary: 2-5 complete sentences, written as a NARRATIVE of the session — what was discussed, what decisions were made and why, what alternatives were considered but not chosen, what got done, what broke. This is what makes the episode different from a plain fact — capture the REASONING and SEQUENCE, not just outcomes.
- key_topics: 1-6 short lowercase topic strings.
- Do not invent details that aren't implied by the input.

Respond ONLY with a valid JSON object. No explanation, no markdown fences.
Example output:
{"title": "Ladakh trip budget planning", "summary": "User planned a 5-day Ladakh trip for 6 friends. Initially considered a relaxed sightseeing itinerary but switched to an extreme-adventure focus after the user expressed interest in intense biking. Settled on a fixed budget of 20000 rupees per person and locked the trip date to August 15th.", "key_topics": ["ladakh trip", "budget", "adventure travel"]}"""

_CRASH_SUMMARY_SYSTEM = """You are a memory summarization assistant.
Given whatever partial record survived from a conversation session that ended abruptly (crash, power loss, forced quit), produce a best-effort title, summary, and key topics from what's available. Note in the summary that the session was interrupted if that's relevant context.

Rules:
- title: max 8 words, no trailing punctuation.
- summary: 1-3 complete sentences describing what was happening based on the available record.
- key_topics: 1-5 short lowercase topic strings.

Respond ONLY with a valid JSON object. No explanation, no markdown fences."""

_MERGE_SUMMARY_SYSTEM = """You are a memory summarization assistant.
Given several older episode summaries from past sessions, merge them into ONE combined title, a 2-4 sentence summary covering the recurring themes, and a list of key topics.

Rules:
- title: max 8 words, no trailing punctuation.
- summary: 2-4 complete sentences capturing the recurring themes across all the episodes, not a list of each one.
- key_topics: 1-6 short lowercase topic strings, deduplicated across the episodes.

Respond ONLY with a valid JSON object. No explanation, no markdown fences."""


def _call_llm_json(system_prompt: str, user_content: str, max_tokens: int = 400) -> dict | None:
    try:
        response = llm_request_lock.post_completion(
            {
                "model": settings.llm_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0.2,
                "max_tokens": max_tokens,
            },
            timeout=_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        raw_text = response.json().get("choices", [{}])[0].get("message", {}).get("content", "{}")
        return _parse_json_object(raw_text)
    except Exception as e:
        log.error("Episodic summarizer LLM call failed: %s", e, exc_info=True)
        return None


def _call_llm_text(system_prompt: str, user_content: str, max_tokens: int = 150) -> str | None:
    """Plain-text variant for summarize_chunk() — a short note, not JSON."""
    try:
        response = llm_request_lock.post_completion(
            {
                "model": settings.llm_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0.2,
                "max_tokens": max_tokens,
            },
            timeout=_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        text = response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        return text or None
    except Exception as e:
        log.error("Chunk summarizer LLM call failed: %s", e, exc_info=True)
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


def summarize_chunk(turns: list[tuple[str, str]]) -> str | None:
    """
    Rolling summary of a small slice of turns (typically 5). Returns a
    plain string, not a dict — this is an intermediate note, not a final
    episode. Returns None if the LLM call fails; callers should skip
    appending a chunk summary rather than storing a fabricated one
    (the raw full_conversation backup still covers this slice either way).
    """
    if not turns:
        return None
    lines = [f"User: {u}\nAssistant: {a}" for u, a in turns if u or a]
    if not lines:
        return None
    return _call_llm_text(_CHUNK_SUMMARY_SYSTEM, "\n".join(lines), max_tokens=150)


def summarize_session(goal, completed_subtasks, memory_updates, last_error, turn_count, chunk_summaries=None) -> dict:
    """Called from session_lifecycle.on_session_end for a clean shutdown.
    Prefers chunk_summaries (rolling notes already generated live) over
    trying to re-summarize the raw transcript, which may be too long to
    fit in one call by session end — that's the whole reason chunking
    happens live rather than only at the end."""
    parts = []
    if chunk_summaries:
        parts.append("Session narrative (in order):")
        for i, s in enumerate(chunk_summaries, 1):
            parts.append(f"  {i}. {s}")
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

    if not chunk_summaries and not goal and not completed_subtasks:
        return fallback

    result = _call_llm_json(_SESSION_SUMMARY_SYSTEM, "\n".join(parts), max_tokens=400)
    return _normalize_result(result, fallback)


def summarize_crashed(session_id, chunk_summaries=None, full_conversation_snippet="", turn_count=0) -> dict:
    """Called from session_lifecycle's crash-recovery sweep. Prefers
    chunk_summaries (rolling notes that survived the crash) over the raw
    full_conversation backup, since chunk summaries are cheaper to feed
    to the LLM and already narrative-shaped. Falls back to the raw
    conversation snippet only if no chunk summary exists yet (e.g. crash
    happened within the first 5 turns, before the first chunk fired)."""
    has_chunks = bool(chunk_summaries)
    has_raw = bool(full_conversation_snippet)

    fallback = {
        "title": "Interrupted session",
        "summary": (
            f"Session {session_id} ended without a clean shutdown after {turn_count} turn(s). "
            + (
                "Recovered narrative: " + " ".join(chunk_summaries)[:400] if has_chunks
                else (f"Last known state: {full_conversation_snippet[:300]}" if has_raw else "No recoverable state.")
            )
        ),
        "key_topics": [],
    }

    if not has_chunks and not has_raw:
        return fallback

    if has_chunks:
        user_content = f"Turn count: {turn_count}\nRecovered session notes (in order):\n" + "\n".join(
            f"  {i}. {s}" for i, s in enumerate(chunk_summaries, 1)
        )
    else:
        user_content = f"Turn count: {turn_count}\nRecovered raw conversation snippet:\n{full_conversation_snippet[:2000]}"

    result = _call_llm_json(_CRASH_SUMMARY_SYSTEM, user_content, max_tokens=300)
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

    result = _call_llm_json(_MERGE_SUMMARY_SYSTEM, "\n".join(lines), max_tokens=400)
    return _normalize_result(result, fallback)