"""
SessionManager/session_lifecycle.py

CHANGED (reflection consolidation): on_session_end now runs a
_consolidate_reflections() pass as its final step, after the episodic
write. It fetches all memory_type='reflection' rows created during this
session, makes ONE background LLM call that judges each directive on the
full session arc, then:

  - Hard-deletes noise rows (low-value, session-specific, or redundant
    with what is already in semantic memory) via
    Database.working_memory_db_client.delete_working_memory().

  - Promotes high-value, cross-session directives to semantic memory
    under category='behavioral_directive' so they survive beyond
    working memory's TTL and remain retrievable even after the
    working_memory row expires.

  - Leaves the rest untouched (cross-session reflections the LLM judged
    as worth keeping but not promoting will live out their expires_at
    and continue to appear in the directive block via
    PromptBuilder.prompt_builder._build_directive_block()).

The consolidation is fully non-fatal: any exception is caught, logged,
and ignored. A failed consolidation never blocks session close, corrupts
other memory layers, or causes the session marker to remain open.

Minimum bar: sessions with fewer than 2 reflection rows skip the LLM
call entirely — there is nothing worth consolidating.
"""

import json
from datetime import datetime, timezone

from MemoryManagement.shortterm_memory.scratchpad import scratchpad
from MemoryManagement.semantic_memory.semantic_memory import semantic_memory
import SessionManager.session_memory_tracker as session_memory_tracker
import Database.active_sessions_db_client as active_sessions_db_client
import Database.working_memory_db_client as working_memory_db_client
import Database.kg_sleep_queue_client as kg_sleep_queue_client
import MemoryManagement.episodic_memory.episodic_memory_store as episodic_memory_store
import MemoryManagement.episodic_memory.summarizer as episodic_summarizer
import LLMEngine.llm_request_lock as llm_request_lock
from GlobalHelpers.config import settings
from GlobalHelpers.logger import get_logger, set_session_id, attach_session_file_handler

log = get_logger(__name__)

# Minimum reflection rows required before we bother making the
# consolidation LLM call. One row alone gives the model nothing to
# compare — there is no "noise vs signal" decision to make.
_MIN_REFLECTIONS_TO_CONSOLIDATE = 2

_CONSOLIDATION_TIMEOUT = 45.0   # seconds — same as reflection_worker


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _render_conversation_snippet(messages: list, max_chars: int = 4000) -> str:
    """Turns the raw {role, content} message list (as saved every turn by
    active_sessions_db_client.save_full_conversation) into a compact
    text snippet for the crash summarizer — used only when NO chunk
    summaries survived (crash happened within the first 5 turns, before
    the first chunk fired)."""
    if not messages:
        return ""
    lines = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content") or ""
        if role in ("user", "assistant") and content:
            lines.append(f"{role}: {content}")
    snippet = "\n".join(lines)
    return snippet[-max_chars:] if len(snippet) > max_chars else snippet


_CONSOLIDATION_SYSTEM = """\
You are Seven's memory consolidation engine.

At session end you review ALL self-correction directives that were produced
during this session and decide which deserve to survive.

You will receive a list of directives, each with an id, text, scope,
specificity, confidence, actionable flag, and novel flag.

For each directive decide one of three outcomes:
  "keep"               — worth keeping in working memory as-is (cross-session
                         or project-level value, not worth promoting to permanent
                         semantic memory yet)
  "delete"             — noise: too session-specific, low-confidence, not
                         actionable, or already captured in semantic memory
  "promote_to_semantic" — high-value, user-level or project-level standing
                         preference that should be permanently remembered even
                         after working memory expires (e.g. "user always wants
                         concise answers", "user prefers bullet lists")

RULES:
- Every directive id in the input must appear in exactly one output list.
- Prefer "keep" over "delete" when uncertain — deleting is permanent.
- Only "promote_to_semantic" directives that are genuinely cross-session
  user preferences or standing behavioural patterns. Do not promote
  session-specific one-offs.
- Respond ONLY with valid JSON. No markdown, no explanation, no preamble.

OUTPUT FORMAT:
{
  "keep":               ["id1", "id2"],
  "delete":             ["id3"],
  "promote_to_semantic": ["id4"]
}"""


def _build_consolidation_user_content(rows: list) -> str:
    """
    Serialises the reflection rows into a compact numbered list for the
    consolidation LLM prompt. Only fields the model needs — id, directive
    text, and the 5 scoring fields. Reasoning is omitted (too verbose).
    """
    lines = [f"Total directives to review: {len(rows)}", ""]
    for i, row in enumerate(rows, 1):
        mem_id = row[0]
        value  = row[3]   # already JSON-decoded by _row_to_tuple
        if not isinstance(value, dict):
            continue
        lines.append(
            f"{i}. id={mem_id}\n"
            f"   directive:   {value.get('directive', '')}\n"
            f"   scope:       {value.get('scope', '?')}\n"
            f"   specificity: {value.get('specificity', '?')}\n"
            f"   confidence:  {value.get('confidence', '?')}\n"
            f"   actionable:  {value.get('actionable', '?')}\n"
            f"   novel:       {value.get('novel', '?')}"
        )
    return "\n".join(lines)


def _consolidate_reflections(session_id: str) -> None:
    """
    Session-end consolidation pass for reflection rows.

    Fetches all memory_type='reflection' rows written during this session,
    makes ONE background LLM call to classify each as keep / delete /
    promote_to_semantic, then executes the decisions.

    Fully non-fatal — any exception at any point is caught and logged.
    Caller (on_session_end) must not depend on this succeeding.

    Skipped entirely when fewer than _MIN_REFLECTIONS_TO_CONSOLIDATE rows
    exist — no LLM call is made and nothing is touched.
    """
    try:
        rows = working_memory_db_client.get_by_type(session_id, "reflection")
    except Exception:
        log.exception("_consolidate_reflections: failed to fetch rows (non-fatal).")
        return

    if not rows:
        log.debug("_consolidate_reflections: no reflection rows for session %s — skipping.", session_id)
        return

    if len(rows) < _MIN_REFLECTIONS_TO_CONSOLIDATE:
        log.info(
            "_consolidate_reflections: only %d reflection row(s) for session %s "
            "— below minimum %d, skipping consolidation LLM call.",
            len(rows), session_id, _MIN_REFLECTIONS_TO_CONSOLIDATE,
        )
        return

    log.info(
        "_consolidate_reflections: reviewing %d reflection(s) for session %s.",
        len(rows), session_id,
    )

    # ----------------------------------------------------------------
    # Build a lookup so we can act on the LLM's decisions by id
    # ----------------------------------------------------------------
    row_by_id = {row[0]: row for row in rows}
    all_ids   = set(row_by_id.keys())

    # ----------------------------------------------------------------
    # LLM call
    # ----------------------------------------------------------------
    user_content = _build_consolidation_user_content(rows)
    try:
        response = llm_request_lock.post_completion(
            {
                "model":    settings.background_llm_model,
                "messages": [
                    {"role": "system", "content": _CONSOLIDATION_SYSTEM},
                    {"role": "user",   "content": user_content},
                ],
                "temperature":          0.1,   # very low — this is a classification task
                "max_tokens":           512,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            role    = "background",
            timeout = _CONSOLIDATION_TIMEOUT,
        )
        response.raise_for_status()
        raw = (
            response.json()
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
    except Exception:
        log.exception(
            "_consolidate_reflections: LLM call failed for session %s (non-fatal).",
            session_id,
        )
        return

    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = "\n".join(
            line for line in raw.splitlines()
            if not line.strip().startswith("```")
        ).strip()

    try:
        decisions = json.loads(raw)
    except json.JSONDecodeError:
        log.warning(
            "_consolidate_reflections: JSON parse failed for session %s — raw: %s",
            session_id, raw[:300],
        )
        return

    if not isinstance(decisions, dict):
        log.warning(
            "_consolidate_reflections: unexpected JSON type %s for session %s.",
            type(decisions), session_id,
        )
        return

    keep_ids    = set(decisions.get("keep",               []) or [])
    delete_ids  = set(decisions.get("delete",             []) or [])
    promote_ids = set(decisions.get("promote_to_semantic",[]) or [])

    # ----------------------------------------------------------------
    # Validate: every input id must appear in exactly one output list.
    # Any id the LLM omitted or put in multiple lists is treated as
    # "keep" (conservative default — deleting is permanent).
    # ----------------------------------------------------------------
    classified  = keep_ids | delete_ids | promote_ids
    duplicates  = (keep_ids & delete_ids) | (keep_ids & promote_ids) | (delete_ids & promote_ids)
    unclassified = all_ids - classified
    bad_ids      = (classified - all_ids) | duplicates   # hallucinated or duplicated

    if bad_ids:
        log.warning(
            "_consolidate_reflections: LLM returned unknown/duplicate ids %s "
            "for session %s — ignoring them.",
            bad_ids, session_id,
        )
        # Remove bad ids from all three buckets; they stay untouched in working_memory
        keep_ids    -= bad_ids
        delete_ids  -= bad_ids
        promote_ids -= bad_ids

    if unclassified:
        log.warning(
            "_consolidate_reflections: LLM omitted %d id(s) %s for session %s "
            "— defaulting to 'keep'.",
            len(unclassified), unclassified, session_id,
        )
        keep_ids |= unclassified

    # ----------------------------------------------------------------
    # Execute: delete
    # ----------------------------------------------------------------
    deleted_count = 0
    for mem_id in delete_ids:
        try:
            ok = working_memory_db_client.delete_working_memory(mem_id)
            if ok:
                deleted_count += 1
                row = row_by_id.get(mem_id)
                directive_text = ""
                if row and isinstance(row[3], dict):
                    directive_text = row[3].get("directive", "")[:60]
                log.info(
                    "_consolidate_reflections: deleted [%s] \"%s\"",
                    mem_id[:8], directive_text,
                )
        except Exception:
            log.exception(
                "_consolidate_reflections: delete failed for id %s (non-fatal).",
                mem_id,
            )

    # ----------------------------------------------------------------
    # Execute: promote_to_semantic
    # ----------------------------------------------------------------
    promoted_count = 0
    for mem_id in promote_ids:
        row = row_by_id.get(mem_id)
        if not row:
            continue
        value = row[3]
        if not isinstance(value, dict):
            continue
        directive_text = str(value.get("directive") or "").strip()
        if not directive_text:
            continue
        confidence = value.get("confidence", 0.5)
        try:
            importance = max(0.4, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            importance = 0.6

        try:
            mem_id_semantic = semantic_memory.store(
                text       = directive_text,
                importance = importance,
                category   = "behavioral_directive",
                source     = "reflection_consolidation",
            )
            if mem_id_semantic:
                promoted_count += 1
                log.info(
                    "_consolidate_reflections: promoted [%s] to semantic memory "
                    "(importance=%.2f): \"%s\"",
                    mem_id[:8], importance, directive_text[:60],
                )
            else:
                log.warning(
                    "_consolidate_reflections: semantic store returned None for [%s].",
                    mem_id[:8],
                )
        except Exception:
            log.exception(
                "_consolidate_reflections: semantic promotion failed for id %s (non-fatal).",
                mem_id,
            )

    log.info(
        "_consolidate_reflections: session %s — kept %d, deleted %d, promoted %d of %d reflection(s).",
        session_id,
        len(keep_ids),
        deleted_count,
        promoted_count,
        len(rows),
    )


def on_session_end(session_id: str) -> None:
    """
    Called when user types /stop, the session exits cleanly, or as the
    shutdown fallback in LLMEngine/cli.py's finally block (Ctrl+C /
    uncaught exception). Promotes important scratchpad state into
    semantic memory, writes a structured session summary into working
    memory, and writes an episodic memory row for this session — then
    wipes the scratchpad.

    Idempotent: LLMEngine/cli.py's clean /stop path and its finally-block
    fallback can both end up calling this for the same session_id (by
    design — the finally block is a safety net, not an alternative
    path). The active_sessions marker is what makes that safe: once a
    session has been closed once, is_session_active() returns False and
    every subsequent call is a fast no-op.
    """
    if not active_sessions_db_client.is_session_active(session_id):
        log.info("on_session_end(%s): session already closed — skipping duplicate call.", session_id)
        return

    # 1. Promote current goal if it exists
    goal = scratchpad.get_current_goal()
    if goal and len(goal) > 10:
        semantic_memory.store(
            text=f"User was working on: {goal}",
            importance=0.75,
            category="goals",
            source="session_end"
        )

    # 2. Promote completed subtasks as experience
    completed_subtasks = scratchpad.get_completed_subtasks()

    for task in completed_subtasks[:3]:
        if task and len(str(task)) > 5:
            semantic_memory.store(
                text=f"User completed task: {task}",
                importance=0.6,
                category="experience",
                source="session_end"
            )

    # 3. Promote any memory_updates the LLM explicitly set
    for update in scratchpad.get_memory_updates():
        # format is "type<insert>:data" or "type<update>:data"
        if isinstance(update, str) and ":" in update:
            _, data = update.split(":", 1)
            if data.strip():
                semantic_memory.store(
                    text=data.strip(),
                    importance=0.65,
                    category="other",
                    source="session_end"
                )

    # 4. Promote last error if it exists (useful for debugging context next session)
    last_error = scratchpad.get_last_error()
    if last_error:
        semantic_memory.store(
            text=f"User encountered an issue last session: {str(last_error)[:200]}",
            importance=0.5,
            category="experience",
            source="session_end"
        )

    # 5. Always persist a structured session summary into working memory
    # too, regardless of whether the LLM used the bridge tool mid-session.
    try:
        import Tools.working_memory_tool as working_memory_tool

        summary_parts = []
        if goal:
            summary_parts.append(f"Goal: {goal}")
        if completed_subtasks:
            summary_parts.append(
                "Completed: " + "; ".join(str(t) for t in completed_subtasks[:5])
            )
        if last_error:
            summary_parts.append(f"Last error: {str(last_error)[:200]}")

        if summary_parts:
            result = working_memory_tool.insert_working_memory(
                memory_type="session_summary",
                key="session_summary",
                value=" | ".join(summary_parts),
                priority=0.6,
                relevance=0.6,
                source="session_end",
            )
            if result is None:
                log.warning("Session-end working memory summary failed to insert.")
    except Exception:
        log.exception("Failed to write session-end working memory summary (non-fatal).")

    # 6. Write the episodic memory row for this session.
    #
    # CHANGED: now stored via episodic_memory_store (Chroma) instead of
    # the old SQLite table, and the summary itself is built from the
    # rolling chunk_summaries that accumulated live during the session
    # (LLMEngine/chunk_summary_worker.py) rather than trying to
    # re-summarize the raw transcript in one call at the end — by
    # session end that transcript may be too long to fit in a single
    # call, which is the whole reason chunking happens live in the
    # first place.
    try:
        turn_count = active_sessions_db_client.get_turn_count(session_id)
        started_at = active_sessions_db_client.get_started_at(session_id) or _now()
        ended_at = _now()
        related_semantic_ids = session_memory_tracker.get_and_clear(session_id)
        chunk_summaries = active_sessions_db_client.get_chunk_summaries(session_id)

        # BUG FIX: summarize_session()'s fallback path always appended
        # "Turn count: N" unconditionally, so its "no notable activity"
        # branch could never actually trigger — every trivial session
        # (immediate /stop or Ctrl+C, turn_count=0) was still writing a
        # real, searchable episode like "Summary: Turn count: 0" into
        # episodic memory, polluting search/recent-episode results with
        # noise. Skip the write entirely below a minimal activity bar.
        if turn_count < 2 and not goal and not completed_subtasks:
            log.info(
                "Session %s had negligible activity (turn_count=%d, no goal, no completed subtasks) — "
                "skipping episodic memory write.",
                session_id, turn_count,
            )
            episode_id = None
        else:
            summary_result = episodic_summarizer.summarize_session(
                goal=goal,
                completed_subtasks=completed_subtasks,
                memory_updates=scratchpad.get_memory_updates(),
                last_error=last_error,
                turn_count=turn_count,
                chunk_summaries=chunk_summaries,
            )

            episode_id = episodic_memory_store.insert_episode(
                session_id=session_id,
                title=summary_result["title"],
                summary=summary_result["summary"],
                key_topics=summary_result.get("key_topics", []),
                start_time_iso=started_at,
                end_time_iso=ended_at,
                turn_count=turn_count,
                outcome=None,  # left unset until a real completion signal exists
                related_semantic_memory_ids=related_semantic_ids,
                importance=0.5,
            )
            if episode_id is None:
                log.warning("Failed to write episodic memory row for session %s.", session_id)
            else:
                # Queue this session for the Knowledge Graph sleep pipeline
                # BEFORE close_session() deletes active_sessions below —
                # this is the only remaining place the conversation text
                # and related_semantic_ids survive past this function.
                try:
                    conversation_text = (
                        "\n\n".join(chunk_summaries) if chunk_summaries
                        else _render_conversation_snippet(
                            active_sessions_db_client.get_full_conversation(session_id)
                        )
                    )
                    kg_sleep_queue_client.enqueue_session(
                        session_id=session_id,
                        episodic_memory_id=episode_id,
                        semantic_memory_ids=related_semantic_ids,
                        conversation_text=conversation_text,
                    )
                except Exception:
                    log.exception(
                        "Failed to enqueue session %s for sleep processing (non-fatal).",
                        session_id,
                    )
    except Exception:
        log.exception("Failed to write episodic memory for session %s (non-fatal).", session_id)
    finally:
        # Close the crash marker regardless of whether the episodic write
        # above succeeded — a failed episodic insert shouldn't leave this
        # session permanently stuck as 'in_progress' and re-swept as a
        # phantom crash on every future startup.
        try:
            active_sessions_db_client.close_session(session_id)
        except Exception:
            log.exception("Failed to close active_sessions marker for session %s (non-fatal).", session_id)

    # 7. Session-end reflection consolidation: review all reflection rows
    #    written during this session, hard-delete noise, promote high-value
    #    directives to semantic memory. Runs AFTER the episodic write so
    #    the episode already exists if we need to reference the session arc.
    #    Fully non-fatal — wrapped internally with its own try/except.
    try:
        _consolidate_reflections(session_id)
    except Exception:
        log.exception(
            "on_session_end: _consolidate_reflections raised unexpectedly for session %s (non-fatal).",
            session_id,
        )

    scratchpad.reset()
    log.info("Session %s ended — scratchpad promoted and cleared.", session_id)


def _recover_stale_sessions(current_session_id: str) -> None:
    """
    Sweeps active_sessions for rows still 'in_progress' that belong to a
    PREVIOUS process (crash, kill -9, power loss — anything that skipped
    on_session_end entirely). Each one is finalized as a best-effort
    'interrupted' episode using whatever chunk_summaries and
    full_conversation backup survived, so that session's history isn't
    silently lost.

    Best-effort throughout: a failure recovering one stale session is
    logged and skipped, never allowed to block startup or the sweep of
    other stale sessions.
    """
    try:
        stale_sessions = active_sessions_db_client.get_stale_sessions(exclude_session_id=current_session_id)
    except Exception:
        log.exception("Failed to query stale sessions for crash recovery (non-fatal).")
        return

    for stale in stale_sessions:
        try:
            _finalize_crashed_session(stale)
        except Exception:
            log.exception(
                "Failed to finalize crashed session %s (non-fatal).",
                stale.get("session_id"),
            )


def _finalize_crashed_session(stale_row: dict) -> None:
    """
    CHANGED: previously fell back to reading working_memory rows (which
    weren't even reliably session-scoped for this purpose). Now uses the
    richer active_sessions columns written live during the session:
      - chunk_summaries: rolling narrative notes, preferred when present
      - full_conversation: raw backup, used only if no chunk exists yet
      - related_semantic_memory_ids: now genuinely populated instead of
        always []  — this is the direct fix for the gap the previous
        version's docstring called an "acceptable loss".
    """
    stale_session_id = stale_row["session_id"]
    turn_count = int(stale_row.get("turn_count") or 0)

    chunk_summaries = stale_row.get("chunk_summaries") or []
    conversation_snippet = ""
    if not chunk_summaries:
        conversation_snippet = _render_conversation_snippet(stale_row.get("full_conversation") or [])

    related_semantic_ids = stale_row.get("related_semantic_memory_ids") or []

    # BUG FIX: same fallback-always-fires issue as on_session_end — skip
    # writing an episode for a session that crashed before anything
    # actually happened (turn_count=0, no chunk summaries, no raw
    # conversation backup at all). A slightly lower bar than the clean
    # -exit gate (turn_count < 1, not < 2) since even one interrupted
    # turn before a crash is worth recording — it's turn_count=0 with
    # nothing recoverable at all that's pure noise.
    if turn_count < 1 and not chunk_summaries and not conversation_snippet:
        log.info(
            "Crashed session %s had no recoverable activity — skipping episodic memory write.",
            stale_session_id,
        )
        active_sessions_db_client.close_session(stale_session_id)
        return

    summary_result = episodic_summarizer.summarize_crashed(
        session_id=stale_session_id,
        chunk_summaries=chunk_summaries,
        full_conversation_snippet=conversation_snippet,
        turn_count=turn_count,
    )

    episode_id = episodic_memory_store.insert_episode(
        session_id=stale_session_id,
        title=summary_result["title"],
        summary=summary_result["summary"],
        key_topics=summary_result.get("key_topics", []),
        start_time_iso=stale_row.get("started_at") or _now(),
        end_time_iso=stale_row.get("last_turn_at") or stale_row.get("started_at") or _now(),
        turn_count=turn_count,
        outcome="interrupted",
        related_semantic_memory_ids=related_semantic_ids,
        importance=0.4,
    )
    if episode_id is None:
        log.warning("Failed to write recovered episodic memory row for crashed session %s.", stale_session_id)
    else:
        log.warning(
            "Recovered crashed session %s as an interrupted episode (%d turns, %d chunk summaries, %d linked facts).",
            stale_session_id, turn_count, len(chunk_summaries), len(related_semantic_ids),
        )
        # Queue the recovered session too — a crash shouldn't leave it
        # permanently invisible to the sleep pipeline just because it
        # never reached a clean on_session_end.
        try:
            conversation_text = (
                "\n\n".join(chunk_summaries) if chunk_summaries else conversation_snippet
            )
            kg_sleep_queue_client.enqueue_session(
                session_id=stale_session_id,
                episodic_memory_id=episode_id,
                semantic_memory_ids=related_semantic_ids,
                conversation_text=conversation_text,
            )
        except Exception:
            log.exception(
                "Failed to enqueue crashed session %s for sleep processing (non-fatal).",
                stale_session_id,
            )

    active_sessions_db_client.close_session(stale_session_id)


def _format_recent_episodes(episodes: list) -> str:
    """Renders the passive 2-episode prefetch into a compact digest for
    the scratchpad. Deliberately short — this is the always-on context
    floor, not a full recall; that's what search_episodic_memory and the
    deterministic trigger are for (see Chunk D)."""
    if not episodes:
        return ""
    lines = ["RECENT SESSIONS:"]
    for ep in episodes:
        title = ep.get("title") or "(untitled)"
        summary = ep.get("summary") or ""
        lines.append(f"  - {title}: {summary}")
    return "\n".join(lines)


def on_session_start(session_id: str, user_query: str = "") -> None:
    """
    Called at the beginning of every session.
    Sweeps for any previous session that crashed without a clean
    shutdown and recovers it, registers THIS session as in-progress, and
    seeds the scratchpad with relevant semantic context AND a passive
    digest of the last 2 episodes so the LLM doesn't start cold.
    """
    # Set the active session id for log lines and create a per-session file
    set_session_id(session_id)
    try:
        attach_session_file_handler(session_id)
    except Exception:
        log.exception("Failed to attach session file handler for %s", session_id)

    # Crash recovery: must run BEFORE this session registers itself,
    # otherwise get_stale_sessions' exclusion filter is irrelevant but
    # ordering still matters for clarity/logging (recover old sessions,
    # then announce the new one).
    try:
        _recover_stale_sessions(current_session_id=session_id)
    except Exception:
        log.exception("Crash-recovery sweep failed at session start (non-fatal).")

    try:
        active_sessions_db_client.start_session(session_id)
    except Exception:
        log.exception("Failed to register active_sessions marker for session %s (non-fatal).", session_id)

    # Pull goals and recent experience to seed context (semantic memory)
    goals = semantic_memory.retrieve_as_text(
        query="what is the user working on", k=2, category="goals"
    )
    experience = semantic_memory.retrieve_as_text(
        query="recent user activity", k=2, category="experience"
    )

    # Passive episodic seed: last 2 episodes, unconditionally — this is
    # the "prefetching summary of previous 2 sessions" floor. Kept small
    # deliberately to keep the always-on token cost low; anything beyond
    # this relies on search_episodic_memory or the deterministic trigger
    # (Chunk D), not this passive seed.
    recent_episodes_digest = ""
    try:
        recent_episodes = episodic_memory_store.get_recent_episodes_capped()
        recent_episodes_digest = _format_recent_episodes(recent_episodes)
    except Exception:
        log.exception("Failed to fetch recent episodes for session start seed (non-fatal).")

    context = "\n".join(filter(None, [goals, experience, recent_episodes_digest]))

    if context:
        # Store as a retrieved memory so it appears in the scratchpad
        scratchpad.add_retrieved_context(context)
    log.info("Session %s started with semantic context.", session_id)