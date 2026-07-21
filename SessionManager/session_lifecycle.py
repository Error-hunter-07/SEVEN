# SessionManager/session_lifecycle.py

from datetime import datetime, timezone

from MemoryManagement.shortterm_memory.scratchpad import scratchpad
from MemoryManagement.semantic_memory.semantic_memory import semantic_memory
import SessionManager.session_memory_tracker as session_memory_tracker
import Database.active_sessions_db_client as active_sessions_db_client
import Database.episodic_memory_db_client as episodic_memory_db_client
import MemoryManagement.episodic_memory.summarizer as episodic_summarizer
from GlobalHelpers.logger import get_logger, set_session_id, attach_session_file_handler

log = get_logger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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

    # 5. BUG FIX: working memory was almost never populated, because it
    # only ever got written if the LLM proactively called
    # add_scratchpad_memory_update mid-conversation — which it rarely did,
    # since the system prompt never explained what working memory is FOR.
    # Now, regardless of whether the LLM used the tool during the session,
    # always persist a structured session summary into working memory too
    # — the same safety-net pattern already used for semantic memory above.
    #
    # This is separate from the semantic-memory promotion: semantic memory
    # holds durable cross-session facts, working memory holds this
    # session's scratch state (goal, what got done, what broke) so a
    # continuation of THIS session (or a quick lookup via
    # get_all_working_memory) has something concrete to read even when
    # the LLM never called the bridge tool itself.
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

    # 6. Write the episodic memory row for this session — a durable,
    # LLM-summarized record of what this session was, separate from both
    # the semantic facts above (durable but atomized) and the working
    # memory summary above (session-scoped, wiped on next session's
    # cleanup elsewhere). This is what makes "what did we do last time"
    # and "how many times have we talked about X" answerable later.
    try:
        turn_count = active_sessions_db_client.get_turn_count(session_id)
        started_at = active_sessions_db_client.get_started_at(session_id) or _now()
        ended_at = _now()
        related_semantic_ids = session_memory_tracker.get_and_clear(session_id)

        summary_result = episodic_summarizer.summarize_session(
            goal=goal,
            completed_subtasks=completed_subtasks,
            memory_updates=scratchpad.get_memory_updates(),
            last_error=last_error,
            turn_count=turn_count,
        )

        episode_id = episodic_memory_db_client.insert_episodic_memory(
            session_id=session_id,
            title=summary_result["title"],
            summary=summary_result["summary"],
            key_topics=summary_result.get("key_topics", []),
            start_time=started_at,
            end_time=ended_at,
            turn_count=turn_count,
            outcome=None,  # left unset until a real completion signal exists
            related_semantic_memory_ids=related_semantic_ids,
            importance=0.5,
        )
        if episode_id is None:
            log.warning("Failed to write episodic memory row for session %s.", session_id)
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

    scratchpad.reset()
    log.info("Session %s ended — scratchpad promoted and cleared.", session_id)


def _recover_stale_sessions(current_session_id: str) -> None:
    """
    Sweeps active_sessions for rows still 'in_progress' that belong to a
    PREVIOUS process (crash, kill -9, power loss — anything that skipped
    on_session_end entirely). Each one is finalized as a best-effort
    'interrupted' episode using whatever durable working_memory rows
    survived, so that session's history isn't silently lost.

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
    stale_session_id = stale_row["session_id"]

    working_memory_snippet = ""
    try:
        import Database.working_memory_db_client as working_memory_db_client
        wm_rows = working_memory_db_client.get_all_current_session_working_memory(stale_session_id) or []
        # _row_to_tuple shape: (id, memory_type, key, value, priority, relevance,
        # created_at, updated_at, expires_at, source, tags) — value is index 3.
        snippets = [str(row[3]) for row in wm_rows if row and row[3] is not None]
        working_memory_snippet = "; ".join(snippets)
    except Exception:
        log.exception("Failed to read working memory for crashed session %s (non-fatal).", stale_session_id)

    turn_count = int(stale_row.get("turn_count") or 0)
    summary_result = episodic_summarizer.summarize_crashed(
        session_id=stale_session_id,
        working_memory_snippets=working_memory_snippet,
        turn_count=turn_count,
    )

    episode_id = episodic_memory_db_client.insert_episodic_memory(
        session_id=stale_session_id,
        title=summary_result["title"],
        summary=summary_result["summary"],
        key_topics=summary_result.get("key_topics", []),
        start_time=stale_row.get("started_at") or _now(),
        end_time=stale_row.get("last_turn_at") or stale_row.get("started_at") or _now(),
        turn_count=turn_count,
        outcome="interrupted",
        related_semantic_memory_ids=[],
        importance=0.4,
    )
    if episode_id is None:
        log.warning("Failed to write recovered episodic memory row for crashed session %s.", stale_session_id)
    else:
        log.warning(
            "Recovered crashed session %s as an interrupted episode (%d turns).",
            stale_session_id, turn_count,
        )

    active_sessions_db_client.close_session(stale_session_id)


def on_session_start(session_id: str, user_query: str = "") -> None:
    """
    Called at the beginning of every session.
    Sweeps for any previous session that crashed without a clean
    shutdown and recovers it, registers THIS session as in-progress, and
    seeds the scratchpad with relevant semantic context so the LLM
    doesn't start cold.
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

     # Pull goals and recent experience to seed context
    goals = semantic_memory.retrieve_as_text(
        query="what is the user working on", k=2, category="goals"
    )
    experience = semantic_memory.retrieve_as_text(
        query="recent user activity", k=2, category="experience"
    )
    context = "\n".join(filter(None, [goals, experience]))

    if context:
        # Store as a retrieved memory so it appears in the scratchpad
        scratchpad.add_retrieved_context(context)
    log.info("Session %s started with semantic context.", session_id)
