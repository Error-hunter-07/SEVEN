"""
LLMEngine/reflection_worker.py

Background worker that produces self-improvement reflections at the end
of every chunk interval (every 5 turns) and optionally on-demand (e.g.
triggered by session_lifecycle.on_session_end for a final consolidated
reflection pass).

WHAT IT DOES
------------
Receives a session snapshot — chunk summaries, current goal, last error,
turn count — and makes ONE background LLM call that produces up to 5
behavioral directives. Each directive is independently scored on 5
criteria and assigned an expires_at duration before being written to
working_memory as memory_type='reflection'.

Because reflections land in working_memory immediately after the worker
finishes, prompt_builder.build_dynamic_context() will pick them up on
the VERY NEXT TURN (it does a live SQLite read every turn via
get_active_reflections_all_sessions). No extra signalling is needed.

ARCHITECTURE
------------
Mirrors LLMEngine/chunk_summary_worker.py exactly:
  - Queue of jobs (not individual turns)
  - One daemon thread
  - contextvars propagated at start() for session-tagged logging
  - flush_and_wait() for orderly shutdown in LLMEngine/cli.py

Unlike extraction_worker, there is NO cooldown/batching here. Each
queued job is its own distinct reflection moment (either every-5-turns
or session-end) and should be processed independently, in order, to
preserve the narrative arc of self-correction across a session.

LLM CALLS
---------
All calls route through llm_request_lock.post_completion(role="background")
— same as episodic summarizer and chunk summary worker — so this worker
never races the main chat turn or the extraction worker for the local
server's single processing slot.

WRITES
------
Writes directly to working_memory_db_client, NOT through
Tools/working_memory_tool.py. The tool layer resolves session_id via
process_manager.get_session_id(), which is a main-thread concept. This
background thread receives session_id explicitly in the job payload and
uses it directly — same pattern as chunk_summary_worker writing to
active_sessions_db_client.

5-CRITERIA SCORING
------------------
The LLM evaluates each directive on:
  1. scope        — 'session' | 'project' | 'user'
  2. specificity  — 'narrow' | 'general'
  3. confidence   — 0.0-1.0 (how strongly the session evidence supports this)
  4. actionable   — bool (does this actually change behaviour, or is it an observation?)
  5. novel        — bool (new insight, or already covered by existing memory?)

These map to a prune_duration_days value which sets expires_at on the
working_memory row. The mapping is intentionally conservative: it is
better to keep a marginally useful directive too long than to lose a
genuinely valuable one too early. Session-end consolidation
(session_lifecycle.on_session_end step 4) will hard-delete noise rows
within the same session, so the expires_at safety net is for cross-session
longevity, not intra-session cleanup.

SCOPE x SPECIFICITY base duration:
  session  + narrow   ->   7 days
  session  + general  ->  14 days
  project  + narrow   ->  45 days
  project  + general  ->  90 days
  user     + narrow   ->  90 days
  user     + general  -> 180 days   (standing behavioural preference)

CONFIDENCE modifier:
  < 0.4  -> halve the base duration
  >= 0.4 -> no change

ACTIONABLE modifier:
  False  -> cap at 14 days regardless of scope (pure observations expire fast)

NOVEL modifier:
  False  -> cap at 7 days (LLM says this is already captured elsewhere)

Final expires_at = now + computed_days.

FALLBACK
--------
If the LLM call fails or returns unparseable JSON, the job is silently
dropped. Reflection is a nice-to-have — a missing reflection never
blocks the session or corrupts any other memory layer.
"""

from __future__ import annotations

import json
import queue
import threading
import contextvars
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Optional

import LLMEngine.llm_request_lock as llm_request_lock
import Database.working_memory_db_client as working_memory_db_client
from GlobalHelpers.config import settings
from GlobalHelpers.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Queue + worker state
# ---------------------------------------------------------------------------

@dataclass
class _ReflectionJob:
    session_id:      str
    chunk_summaries: list          # rolling narrative notes from this session so far
    goal:            Optional[str] # scratchpad current_goal
    last_error:      Optional[str] # scratchpad last_error
    turn_count:      int
    trigger:         str           # 'chunk' | 'session_end' -- logged, not used in logic


_reflection_queue: "queue.Queue[_ReflectionJob]" = queue.Queue()
_worker_lock    = threading.Lock()
_worker_started = False

# flush_and_wait() support -- mirrors extraction_worker.py
_flush_requested = threading.Event()
_drained         = threading.Event()
_drained.set()   # nothing pending at import time

_REQUEST_TIMEOUT = 45  # slightly longer than summarizer -- scoring is heavier

# ---------------------------------------------------------------------------
# Duration scoring
# ---------------------------------------------------------------------------

_BASE_DURATION = {
    ("session", "narrow"):   7,
    ("session", "general"):  14,
    ("project", "narrow"):   45,
    ("project", "general"):  90,
    ("user",    "narrow"):   90,
    ("user",    "general"):  180,
}

_SCOPE_VALUES    = {"session", "project", "user"}
_SPECIFIC_VALUES = {"narrow", "general"}


def _compute_expires_at(
    scope:       str,
    specificity: str,
    confidence:  float,
    actionable:  bool,
    novel:       bool,
) -> str:
    """
    Applies the 5-criteria scoring matrix and returns an ISO-format
    expires_at string. Always returns a valid string -- falls back to
    14 days on any bad input so a malformed LLM response never crashes
    the insert.
    """
    scope       = scope.lower()       if scope       in _SCOPE_VALUES    else "session"
    specificity = specificity.lower() if specificity in _SPECIFIC_VALUES else "narrow"
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    base_days = _BASE_DURATION.get((scope, specificity), 14)
    days      = base_days

    if not actionable:
        days = min(days, 14)        # pure observations expire fast

    if not novel:
        days = min(days, 7)         # duplicate knowledge pruned fast

    if confidence < 0.4:
        days = max(3, days // 2)    # weak evidence -> halve, floor at 3 days

    expires_at = (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()
    log.debug(
        "_compute_expires_at: scope=%s specificity=%s confidence=%.2f "
        "actionable=%s novel=%s -> %d days -> %s",
        scope, specificity, confidence, actionable, novel, days, expires_at,
    )
    return expires_at


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

_REFLECTION_SYSTEM = """\
You are Seven's internal self-improvement engine.

Given a snapshot of what just happened in a conversation session, produce
up to 5 behavioral directives -- concrete, actionable changes Seven should
make in future turns or sessions.

For each directive you MUST score it on 5 criteria:
  scope        -- one of: "session" | "project" | "user"
                 session  = only relevant for the rest of this single session
                 project  = relevant for an ongoing project the user is working on
                 user     = a standing preference or pattern that transcends any project
  specificity  -- one of: "narrow" | "general"
                 narrow   = applies to one specific situation or topic
                 general  = applies broadly across many future interactions
  confidence   -- float 0.0-1.0: how strongly does THIS session's evidence support
                 the directive? 0.3 = weak hint, 0.7 = clear pattern, 0.9+ = explicit feedback
  actionable   -- true if this changes a concrete behaviour; false if it is just an observation
  novel        -- true if this is new insight; false if it likely duplicates an existing preference

RULES:
- Only produce directives supported by THIS snapshot -- do not invent.
- Directives must be concrete: "Ask for budget before planning" not "Be more helpful".
- If there is nothing worth reflecting on, return an empty directives list.
- Respond ONLY with valid JSON. No markdown, no explanation, no preamble.

OUTPUT FORMAT:
{
  "directives": [
    {
      "directive": "...",
      "reasoning": "...",
      "scope": "project",
      "specificity": "general",
      "confidence": 0.75,
      "actionable": true,
      "novel": true
    }
  ]
}"""


def _build_user_content(job: _ReflectionJob) -> str:
    parts = [f"Turn count: {job.turn_count}", f"Trigger: {job.trigger}"]
    if job.goal:
        parts.append(f"Goal: {job.goal}")
    if job.chunk_summaries:
        parts.append("Session narrative so far (in order):")
        for i, s in enumerate(job.chunk_summaries, 1):
            parts.append(f"  {i}. {s}")
    if job.last_error:
        parts.append(f"Last error: {str(job.last_error)[:300]}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# LLM call + parse
# ---------------------------------------------------------------------------

def _call_llm(job: _ReflectionJob):
    """
    Makes ONE background LLM call and returns a list of raw directive
    dicts, or None if the call or parse fails. Callers treat None as
    "skip this job silently".
    """
    user_content = _build_user_content(job)
    try:
        response = llm_request_lock.post_completion(
            {
                "model":    settings.background_llm_model,
                "messages": [
                    {"role": "system", "content": _REFLECTION_SYSTEM},
                    {"role": "user",   "content": user_content},
                ],
                "temperature":          0.3,
                "max_tokens":           600,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            role    = "background",
            timeout = _REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        raw = (
            response.json()
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
    except Exception as e:
        log.error("reflection_worker LLM call failed: %s", e, exc_info=True)
        return None

    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = "\n".join(
            line for line in raw.splitlines()
            if not line.strip().startswith("```")
        ).strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        log.warning("reflection_worker JSON parse failed: %s -- raw: %s", e, raw[:300])
        return None

    if not isinstance(parsed, dict):
        log.warning("reflection_worker: top-level JSON is not an object -- got %s", type(parsed))
        return None

    directives = parsed.get("directives")
    if not isinstance(directives, list):
        log.warning("reflection_worker: 'directives' key missing or not a list")
        return None

    return directives


# ---------------------------------------------------------------------------
# Write to working_memory
# ---------------------------------------------------------------------------

def _write_reflections(session_id: str, directives: list) -> int:
    """
    Writes each valid directive as a working_memory row with
    memory_type='reflection'. Returns the count of rows successfully
    inserted.

    Invalid or empty directive dicts are skipped with a warning rather
    than aborting the whole batch -- a partially bad LLM response should
    still persist whatever was good.
    """
    written = 0
    for raw in directives:
        if not isinstance(raw, dict):
            log.warning("reflection_worker: skipping non-dict directive: %s", raw)
            continue

        directive_text = str(raw.get("directive") or "").strip()
        if not directive_text:
            log.warning("reflection_worker: skipping directive with empty text")
            continue

        reasoning   = str(raw.get("reasoning") or "").strip()
        scope       = raw.get("scope",       "session")
        specificity = raw.get("specificity", "narrow")
        confidence  = raw.get("confidence",  0.5)
        actionable  = bool(raw.get("actionable", True))
        novel       = bool(raw.get("novel",       True))

        expires_at = _compute_expires_at(scope, specificity, confidence, actionable, novel)

        # priority mirrors confidence so get_active_reflections_all_sessions()
        # surfaces the most confident directives first
        try:
            priority = max(0.1, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            priority = 0.5

        value = {
            "directive":   directive_text,
            "reasoning":   reasoning,
            "scope":       scope,
            "specificity": specificity,
            "confidence":  confidence,
            "actionable":  actionable,
            "novel":       novel,
        }

        mem_id = working_memory_db_client.insert_working_memory(
            session_id  = session_id,
            memory_type = "reflection",
            key         = "directive",
            value       = value,
            priority    = priority,
            relevance   = priority,       # same as priority -- no separate signal yet
            source      = "reflection_worker",
            expires_at  = expires_at,
        )

        if mem_id:
            written += 1
            log.info(
                "Reflection written [%s] scope=%s confidence=%.2f expires=%s: %s",
                mem_id[:8], scope, confidence, expires_at[:10], directive_text[:80],
            )
        else:
            log.warning(
                "reflection_worker: insert_working_memory returned None for directive: %s",
                directive_text[:80],
            )

    return written


# ---------------------------------------------------------------------------
# Worker loop
# ---------------------------------------------------------------------------

def _process_job(job: _ReflectionJob) -> None:
    """
    Full pipeline for one reflection job: LLM call -> parse -> write.
    Non-fatal throughout -- any exception is logged and the job is dropped.
    """
    log.debug(
        "reflection_worker: processing job trigger=%s session=%s turns=%d chunks=%d",
        job.trigger, job.session_id, job.turn_count, len(job.chunk_summaries),
    )

    directives = _call_llm(job)
    if directives is None:
        log.warning(
            "reflection_worker: LLM call returned nothing for session %s (trigger=%s) -- skipping.",
            job.session_id, job.trigger,
        )
        return

    if not directives:
        log.info(
            "reflection_worker: LLM found nothing worth reflecting on "
            "(session=%s trigger=%s) -- 0 directives written.",
            job.session_id, job.trigger,
        )
        return

    written = _write_reflections(job.session_id, directives)
    log.info(
        "reflection_worker: wrote %d/%d directive(s) for session %s (trigger=%s).",
        written, len(directives), job.session_id, job.trigger,
    )


def _reflection_worker() -> None:
    while True:
        job = _reflection_queue.get()
        _drained.clear()
        try:
            _process_job(job)
        except Exception:
            log.exception(
                "reflection_worker: unhandled exception processing job "
                "(session=%s trigger=%s) -- non-fatal, continuing.",
                job.session_id, job.trigger,
            )
        finally:
            _reflection_queue.task_done()
            if _reflection_queue.empty():
                _drained.set()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def queue_reflection_job(
    session_id:      str,
    chunk_summaries: list,
    goal:            Optional[str] = None,
    last_error:      Optional[str] = None,
    turn_count:      int           = 0,
    trigger:         str           = "chunk",
) -> None:
    """
    Enqueue a reflection job. Non-blocking -- returns immediately.

    trigger='chunk'       -- called by llm_client.py every CHUNK_INTERVAL_TURNS
                             (same moment chunk_summary_worker.queue_chunk fires)
    trigger='session_end' -- called by session_lifecycle.on_session_end for a
                             final reflection pass over the whole session narrative

    chunk_summaries should be the FULL list accumulated so far this session
    (from active_sessions_db_client.get_chunk_summaries), not just the latest
    chunk -- the LLM needs the arc to judge scope and novelty accurately.
    """
    if not session_id:
        log.warning("queue_reflection_job: called with empty session_id -- skipping.")
        return
    _drained.clear()
    job = _ReflectionJob(
        session_id      = session_id,
        chunk_summaries = list(chunk_summaries) if chunk_summaries else [],
        goal            = goal,
        last_error      = last_error,
        turn_count      = turn_count,
        trigger         = trigger,
    )
    _reflection_queue.put(job)
    log.debug(
        "queue_reflection_job: enqueued trigger=%s session=%s.", trigger, session_id
    )


def flush_and_wait(timeout: float = 60.0) -> bool:
    """
    Blocks until the reflection queue is fully drained or `timeout` seconds
    elapse. Call this from LLMEngine/cli.py's shutdown path BEFORE stopping
    the background LLM server, so any queued jobs can still make their LLM
    call.

    Returns True if drained before timeout, False otherwise (logged as a
    warning, never treated as fatal).
    """
    if _reflection_queue.empty() and _drained.is_set():
        return True                 # nothing pending -- fast path
    finished = _drained.wait(timeout=timeout)
    if not finished:
        log.warning(
            "reflection_worker.flush_and_wait timed out after %.0fs -- "
            "some pending reflections may not have been written before shutdown.",
            timeout,
        )
    return finished


def start() -> None:
    """
    Starts the background worker thread exactly once. Propagates the
    current contextvars so log lines from this thread stay tagged with
    the right session_id.

    Called from LLMEngine/llm_client.py's bootstrap block alongside
    extraction_worker.start() and chunk_summary_worker.start().
    """
    global _worker_started
    with _worker_lock:
        if _worker_started:
            return
        ctx = contextvars.copy_context()
        threading.Thread(
            target = lambda: ctx.run(_reflection_worker),
            daemon = True,
            name   = "ReflectionWorker",
        ).start()
        _worker_started = True
        log.info("reflection_worker: background thread started.")