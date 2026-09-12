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
                          LLMEngine/chunk_summary_worker.py. A rough
                          narrative note, not a polished summary, since
                          it gets compressed again by summarize_session()
                          at session end. max_tokens=1400 — generous on
                          purpose (see the "Fix" note this file already
                          carried: a too-low max_tokens previously
                          produced implausibly short summaries in
                          production). Retries once with a stronger
                          prompt, then falls back to a cheap extractive
                          note, if the model's output looks too thin for
                          how much source material it was given — see
                          _looks_too_thin() below. summarize_session()/
                          summarize_crashed() below still cap how many
                          of these get concatenated into one prompt (see
                          _build_chunk_narrative()), so a long session
                          with many chunks still can't silently overflow
                          the background model's 8192-token context.
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
# Fix - becoz the LLM was not getting proper context and was getting very short summaries, the quality of the knowledge graph generated was very poor, so increased the max tokens
# and also few changes to the prompts so as to get proper summaries

from __future__ import annotations

import json

from GlobalHelpers.logger import get_logger
from GlobalHelpers.config import settings
import LLMEngine.llm_request_lock as llm_request_lock

log = get_logger(__name__)

_REQUEST_TIMEOUT = 30

_CHUNK_SUMMARY_SYSTEM = """You are a memory summarization assistant.
Given a short slice of a conversation (a few turns), write a note of 4-8 sentences (at least 60 words) capturing what happened in THIS slice: what was discussed, decided, or done. This is an intermediate note, not a final summary — it will be compressed again later, so favor concrete, checkable facts over vague reflection.

You MUST:
- Name every specific person, place, object, or named entity that appears in the turns below (do not write generic phrases like "the story continues" instead of the actual names).
- Include any concrete decisions made, specific numbers, and options that were considered but rejected.
- Cover the SLICE AS A WHOLE, in order — not just the most recent turn. If five turns are given, your note must reflect content from across all five, not only the last one.

Do not invent anything that isn't in the conversation. Do not repeat the same sentence or phrase more than once. A short, vague, or generic note is a failed note — be specific.
Respond with ONLY the note text. No JSON, no markdown, no preamble."""

# A retry system prompt used the ONE time the first attempt comes back
# implausibly short/vague for how much source material it was given (see
# _looks_too_thin() below). Explicitly naming the failure mode gets a
# small model to actually correct it, rather than repeating the same
# generic non-answer a second time.
_CHUNK_SUMMARY_RETRY_SYSTEM = _CHUNK_SUMMARY_SYSTEM + """

Your previous attempt at this exact task was rejected for being too short/vague — it read something like generic filler ("the story continues", "they discussed various things") instead of naming the actual people, places, objects, and events from the conversation. Do not repeat that mistake. Write the specific note now."""

# Below this word count, a chunk summary is almost certainly the "generic
# one-liner" failure mode seen in production (a 5-turn, several-hundred-
# word slice compressed to a single vague sentence with zero named
# entities) rather than a genuinely uneventful slice. Used to trigger one
# retry with a more explicit prompt before falling back to an extractive
# note — see summarize_chunk() below.
_MIN_CHUNK_SUMMARY_WORDS = 25

_SESSION_SUMMARY_SYSTEM = """You are a memory summarization assistant.
Given a sequence of notes describing what happened across a conversation session (in order), plus some structured signals, produce a title, a summary, and key topics.

Rules:
- title: max 8 words, no trailing punctuation.
- summary: 10 - 12 complete sentences, written as a NARRATIVE of the session — what was discussed, what decisions were made and why, what alternatives were considered but not chosen, what got done, what broke. This is what makes the episode different from a plain fact — capture the REASONING and SEQUENCE, not just outcomes. Don't make it much short, keep the content as much as possible. Make sure there is no data loss from the content given to you.
- key_topics: 1-6 short lowercase topic strings.
- Do not invent details that aren't implied by the input.

Respond ONLY with a valid JSON object. No explanation, no markdown fences.
Example output:
{"title": "Ladakh trip budget planning", "summary": "User planned a 5-day Ladakh trip for 6 friends. Initially considered a relaxed sightseeing itinerary but switched to an extreme-adventure focus after the user expressed interest in intense biking. Settled on a fixed budget of 20000 rupees per person and locked the trip date to August 15th.", "key_topics": ["ladakh trip", "budget", "adventure travel"]}"""

_CRASH_SUMMARY_SYSTEM = """You are a memory summarization assistant.
Given whatever partial record survived from a conversation session that ended abruptly (crash, power loss, forced quit), produce a best-effort title, summary, and key topics from what's available. Note in the summary that the session was interrupted if that's relevant context. Don't make it much short, keep the content as much as possible. Make sure there is no data loss from the content given to you.

Rules:
- title: max 8 words, no trailing punctuation.
- summary: 10 - 12 complete sentences describing what was happening based on the available record.
- key_topics: 1-5 short lowercase topic strings.

Respond ONLY with a valid JSON object. No explanation, no markdown fences."""

_MERGE_SUMMARY_SYSTEM = """You are a memory summarization assistant.
Given several older episode summaries from past sessions, merge them into ONE combined title, a 2-4 sentence summary covering the recurring themes, and a list of key topics. Don't make it much short, keep the content as much as possible. Make sure there is no data loss from the content given to you.

Rules:
- title: max 8 words, no trailing punctuation.
- summary: 10 - 12 complete sentences capturing the recurring themes across all the episodes, not a list of each one.
- key_topics: 1-6 short lowercase topic strings, deduplicated across the episodes.

Respond ONLY with a valid JSON object. No explanation, no markdown fences."""


import re

_REASONING_BLOCK = re.compile(
    r"<(think|thinking|reasoning)>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)


def _strip_hidden_reasoning(text: str | None) -> str | None:
    """
    Defense-in-depth against chain-of-thought leaking into what's
    supposed to be a plain summary/JSON response. The real fix is
    Runtime/process_manager.py's --jinja flag (without it,
    chat_template_kwargs={"enable_thinking": False} is silently
    ignored by llama-server) — this is a second layer in case a given
    model/template still emits visible <think>/<thinking>/<reasoning>
    tags regardless, or in case --jinja hasn't been deployed yet.
    Logs a warning with the approximate token cost when it strips
    something, since that number is exactly what explains "the model
    gives the same short answer no matter what the prompt says": the
    reasoning ate the max_tokens budget before real content began.
    """
    if not text:
        return text
    match = _REASONING_BLOCK.search(text)
    if not match:
        return text
    stripped = _REASONING_BLOCK.sub("", text).strip()
    log.warning(
        "Stripped a hidden reasoning block from LLM output (~%d chars / "
        "~%d tokens) — if this appears often, chat_template_kwargs' "
        "enable_thinking=False isn't taking effect for this model; see "
        "Runtime/process_manager.py's --jinja flag. Remaining content "
        "(%d words): %r",
        len(match.group(0)), len(match.group(0)) // 4,
        _word_count(stripped), stripped[:200],
    )
    return stripped


def _call_llm_json(system_prompt: str, user_content: str, max_tokens: int = 1600) -> dict | None:
    try:
        response = llm_request_lock.post_completion(
            {
                "model": settings.background_llm_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0.1,
                "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            role="background",
            timeout=_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        raw_text = response.json().get("choices", [{}])[0].get("message", {}).get("content") or "{}"
        raw_text = _strip_hidden_reasoning(raw_text) or "{}"
        return _parse_json_object(raw_text)
    except Exception as e:
        log.error("Episodic summarizer LLM call failed: %s", e, exc_info=True)
        return None


def _call_llm_text(system_prompt: str, user_content: str, max_tokens: int = 1600) -> str | None:
    """Plain-text variant for summarize_chunk() — a short note, not JSON."""
    try:
        response = llm_request_lock.post_completion(
            {
                "model": settings.background_llm_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0.1,
                "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            role="background",
            timeout=_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        text = response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        text = _strip_hidden_reasoning(text)
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


# Rough char budget for the chunk-summary narrative block handed to the
# background model. The background server runs with ctx_size=8192 (see
# Runtime/background_process.py) shared between the system prompt, this
# narrative, any other structured signals, AND the completion's
# max_tokens (up to 1600 for summarize_session/summarize_crashed). ~4
# chars/token is a conservative estimate for English text, so this stays
# comfortably under budget even on longer sessions instead of relying on
# llama-server to fail gracefully (or context-shift and silently drop
# older content) when the prompt runs too long.
_CHUNK_NARRATIVE_CHAR_BUDGET = 12000


def _build_chunk_narrative(chunk_summaries: list[str]) -> tuple[list[str], bool]:
    """
    Returns (numbered narrative lines, was_truncated). Keeps the MOST
    RECENT chunk summaries within _CHUNK_NARRATIVE_CHAR_BUDGET rather than
    the earliest ones — the end of a session is usually what a "what
    happened" summary most needs to get right, and silently dropping the
    tail (as an unbounded prompt would risk via context-shift) is worse.
    """
    if not chunk_summaries:
        return [], False

    kept: list[str] = []
    total_chars = 0
    truncated = False
    for s in reversed(chunk_summaries):
        total_chars += len(s)
        if total_chars > _CHUNK_NARRATIVE_CHAR_BUDGET and kept:
            truncated = True
            break
        kept.append(s)
    kept.reverse()

    offset = len(chunk_summaries) - len(kept)
    lines = [f"  {i + offset + 1}. {s}" for i, s in enumerate(kept)]
    if truncated:
        lines.insert(0, f"  (earliest {offset} chunk note(s) omitted — over length budget)")
    return lines, truncated


def _word_count(text: str) -> int:
    return len(text.split())


def _looks_too_thin(summary: str, source_lines: list[str]) -> bool:
    """
    Heuristic for the exact failure mode seen in production: a 5-turn,
    several-hundred-word slice (multiple named characters, concrete plot
    events) compressed into a single vague sentence with no specifics.
    Deliberately cheap/string-based (no LLM call) — this just decides
    whether it's worth spending a retry, not whether the content is
    "good" in any deep sense.
    """
    if not summary:
        return True
    source_words = sum(_word_count(l) for l in source_lines)
    # Only flag slices that actually had enough source material that a
    # near-empty summary is implausible — a genuinely short/quiet slice
    # (e.g. two one-line turns) legitimately earns a short note.
    if source_words < 40:
        return False
    return _word_count(summary) < _MIN_CHUNK_SUMMARY_WORDS


def _extractive_fallback(turns: list[tuple[str, str]]) -> str:
    """
    Last-resort note when the LLM produces thin output twice in a row
    (initial attempt + retry). Not a real summary — just the first
    sentence of each user message and each assistant reply, stitched
    together — but it preserves actual names/nouns from the
    conversation instead of storing another vague LLM sentence that
    entity extraction can do nothing with. Better than empty; still
    clearly inferior to the LLM path, which is why this only runs after
    two failed LLM attempts.
    """
    import re
    parts: list[str] = []
    for u, a in turns:
        for speaker, text in (("User", u), ("Assistant", a)):
            if not text:
                continue
            first_sentence = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)[0]
            if first_sentence:
                parts.append(f"{speaker}: {first_sentence.strip()[:200]}")
    return " ".join(parts)[:1500] or None


def summarize_chunk(turns: list[tuple[str, str]]) -> str | None:
    """
    Rolling summary of a small slice of turns (typically 5). Returns a
    plain string, not a dict — this is an intermediate note, not a final
    episode. Returns None if the LLM call fails; callers should skip
    appending a chunk summary rather than storing a fabricated one
    (the raw full_conversation backup still covers this slice either way).

    Retries once with a more explicit prompt if the first attempt looks
    implausibly thin for how much source material it was given (see
    _looks_too_thin()), then falls back to a cheap extractive note if the
    retry is also thin — a chunk summary this app relies on as the
    "primary extraction signal" for the knowledge graph must never
    silently collapse into an uninformative one-liner.
    """
    if not turns:
        return None
    lines = [f"User: {u}\nAssistant: {a}" for u, a in turns if u or a]
    if not lines:
        return None

    user_content = "\n".join(lines)

    # max_tokens=1400: bumped up from 900 as extra headroom on top of the
    # --jinja fix in Runtime/process_manager.py (which is what actually
    # makes enable_thinking=False take effect). Belt-and-suspenders: if
    # some hidden reasoning still leaks through for a given model/
    # template, this leaves more room for real content after it, and
    # _strip_hidden_reasoning() above removes any <think> block that
    # does show up in the final text either way. Still comfortably
    # inside the background model's 8192-token context for a 5-turn slice.
    summary = _call_llm_text(_CHUNK_SUMMARY_SYSTEM, user_content, max_tokens=1400)
    log.debug("summarize_chunk: first attempt (%d words): %r",
              _word_count(summary or ""), (summary or "")[:300])

    if _looks_too_thin(summary or "", lines):
        log.warning(
            "summarize_chunk: first attempt looks too thin (%d word(s) for "
            "%d source line(s)) — retrying with a more explicit prompt.",
            _word_count(summary or ""), len(lines),
        )
        retry = _call_llm_text(_CHUNK_SUMMARY_RETRY_SYSTEM, user_content, max_tokens=1400)
        log.debug("summarize_chunk: retry attempt (%d words): %r",
                  _word_count(retry or ""), (retry or "")[:300])
        if retry and not _looks_too_thin(retry, lines):
            summary = retry
        elif retry:
            # Retry didn't help — prefer the extractive fallback over
            # either thin LLM attempt so at least real names/nouns from
            # the conversation survive into the stored chunk summary.
            log.warning(
                "summarize_chunk: retry also looked thin — falling back "
                "to an extractive note for this chunk."
            )
            summary = _extractive_fallback(turns) or summary or retry

    return summary


def summarize_session(goal, completed_subtasks, memory_updates, last_error, turn_count, chunk_summaries=None) -> dict:
    """Called from session_lifecycle.on_session_end for a clean shutdown.
    Prefers chunk_summaries (rolling notes already generated live) over
    trying to re-summarize the raw transcript, which may be too long to
    fit in one call by session end — that's the whole reason chunking
    happens live rather than only at the end."""
    parts = []
    if chunk_summaries:
        parts.append("Session narrative (in order):")
        narrative_lines, _truncated = _build_chunk_narrative(chunk_summaries)
        parts.extend(narrative_lines)
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

    result = _call_llm_json(_SESSION_SUMMARY_SYSTEM, "\n".join(parts), max_tokens=1600)
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
                "Recovered narrative: " + " ".join(chunk_summaries)[:2000] if has_chunks
                else (f"Last known state: {full_conversation_snippet[:4000]}" if has_raw else "No recoverable state.")
            )
        ),
        "key_topics": [],
    }

    if not has_chunks and not has_raw:
        return fallback

    if has_chunks:
        narrative_lines, _truncated = _build_chunk_narrative(chunk_summaries)
        user_content = f"Turn count: {turn_count}\nRecovered session notes (in order):\n" + "\n".join(narrative_lines)
    else:
        user_content = f"Turn count: {turn_count}\nRecovered raw conversation snippet:\n{full_conversation_snippet[:2000]}"

    result = _call_llm_json(_CRASH_SUMMARY_SYSTEM, user_content, max_tokens=1600)
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
        "summary": " ".join(str(ep.get("summary") or "") for ep in episodes)[:2000] or "No summary available.",
        "key_topics": sorted({t for ep in episodes for t in (ep.get("key_topics") or [])})[:6],
    }

    if not lines:
        return fallback

    result = _call_llm_json(_MERGE_SUMMARY_SYSTEM, "\n".join(lines), max_tokens=1600)
    return _normalize_result(result, fallback)