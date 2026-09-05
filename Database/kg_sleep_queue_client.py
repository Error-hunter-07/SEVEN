"""
Database/kg_sleep_queue_client.py

CRUD for the kg_sleep_queue table — the durable hand-off point between
a session ending and the Knowledge Graph sleep pipeline processing it.

  enqueue_session()      called from SessionManager/session_lifecycle.py
                          (on_session_end and _finalize_crashed_session)
                          BEFORE active_sessions_db_client.close_session()
                          deletes the session's raw context.
  get_pending_sessions()  called by the sleep pipeline (memory_selector.py)
                          to pull its next batch of work, oldest first.
  mark_processed()        called by the sleep pipeline once a session's
                          entities/edges have been extracted.
  count_pending()         "how many sessions are waiting to be processed" —
                          used for progress reporting (e.g. /sleep status).
  delete_processed()      periodic cleanup of old, already-processed rows.

See Database/local_db.py's _ensure_kg_sleep_queue_schema() for the full
column-by-column rationale.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Optional

import Database.local_db as local_db
from Database.kg_constants import _now
from GlobalHelpers.logger import get_logger

log = get_logger(__name__)


def _row_to_queue_entry(row) -> dict:
    """
    Convert a sqlite3.Row from kg_sleep_queue into a plain dict.

    QUEUE ENTRY DICT SHAPE:
      {
        "session_id":          str,
        "episodic_memory_id":  str,
        "semantic_memory_ids": list[str],  # decoded from JSON
        "conversation_text":   str,
        "queued_at":           str,   # ISO timestamp
        "processed_at":        str | None,  # ISO timestamp, None if pending
      }
    """
    try:
        semantic_memory_ids = (
            json.loads(row["semantic_memory_ids"]) if row["semantic_memory_ids"] else []
        )
    except (json.JSONDecodeError, TypeError):
        semantic_memory_ids = []

    return {
        "session_id":          row["session_id"],
        "episodic_memory_id":  row["episodic_memory_id"],
        "semantic_memory_ids": semantic_memory_ids,
        "conversation_text":   row["conversation_text"] or "",
        "queued_at":           row["queued_at"],
        "processed_at":        row["processed_at"],
    }


def enqueue_session(
    session_id:           str,
    episodic_memory_id:   str,
    semantic_memory_ids:  Optional[list[str]] = None,
    conversation_text:    str                 = "",
) -> bool:
    """
    Write (or refresh) this session's row in kg_sleep_queue.

    Must be called BEFORE active_sessions_db_client.close_session()
    deletes the active_sessions row for this session — this is the only
    place that context gets copied into something the sleep pipeline
    can still read after the session is gone.

    session_id is the PRIMARY KEY. If a row already exists for this
    session (e.g. a retried on_session_end after a partial failure),
    ON CONFLICT overwrites it with the fresh values and resets
    processed_at back to NULL — the safest behaviour, since a re-run
    means the sleep pipeline should look at this session again.

    Returns True on success, False on missing required fields or error.
    """
    if not session_id or not episodic_memory_id:
        log.warning(
            "enqueue_session: missing session_id or episodic_memory_id — skipping."
        )
        return False

    conn = local_db.get_connection()
    now = _now()
    try:
        conn.execute(
            """
            INSERT INTO kg_sleep_queue
                (session_id, episodic_memory_id, semantic_memory_ids,
                 conversation_text, queued_at, processed_at)
            VALUES (?, ?, ?, ?, ?, NULL)
            ON CONFLICT(session_id) DO UPDATE SET
                episodic_memory_id  = excluded.episodic_memory_id,
                semantic_memory_ids = excluded.semantic_memory_ids,
                conversation_text   = excluded.conversation_text,
                queued_at           = excluded.queued_at,
                processed_at        = NULL
            """,
            (
                session_id,
                episodic_memory_id,
                json.dumps(semantic_memory_ids or []),
                conversation_text or "",
                now,
            ),
        )
        conn.commit()
        log.info(
            "enqueue_session: queued session %s for sleep processing "
            "(episode=%s, %d semantic ids).",
            session_id, episodic_memory_id, len(semantic_memory_ids or []),
        )
        return True
    except Exception as e:
        conn.rollback()
        log.error("enqueue_session(%s) error: %s", session_id, e, exc_info=True)
        return False


def get_pending_sessions(limit: int = 10) -> list[dict]:
    """
    Return up to `limit` unprocessed sessions, oldest queued first —
    the sleep pipeline's next batch of work.

    Returns [] on empty result or error.
    """
    conn = local_db.get_connection()
    try:
        rows = conn.execute(
            """
            SELECT session_id, episodic_memory_id, semantic_memory_ids,
                   conversation_text, queued_at, processed_at
            FROM kg_sleep_queue
            WHERE processed_at IS NULL
            ORDER BY queued_at ASC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
        return [_row_to_queue_entry(r) for r in rows]
    except Exception as e:
        log.error("get_pending_sessions error: %s", e, exc_info=True)
        return []


def get_session_entry(session_id: str) -> Optional[dict]:
    """
    Fetch a single session's queue row, regardless of processed state.
    Returns None if no row exists or on error.
    """
    if not session_id:
        return None
    conn = local_db.get_connection()
    try:
        row = conn.execute(
            """
            SELECT session_id, episodic_memory_id, semantic_memory_ids,
                   conversation_text, queued_at, processed_at
            FROM kg_sleep_queue
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
        return _row_to_queue_entry(row) if row else None
    except Exception as e:
        log.error("get_session_entry(%s) error: %s", session_id, e, exc_info=True)
        return None


def mark_processed(session_id: str) -> bool:
    """
    Stamp a session's row as processed (processed_at = now) once the
    sleep pipeline has finished extracting entities/edges from it.

    Rows are stamped, not deleted, so a completed batch stays
    inspectable — see delete_processed() for cleanup.

    Returns True if a row was updated, False if no matching row exists
    or on error.
    """
    if not session_id:
        log.warning("mark_processed: empty session_id — skipping.")
        return False

    conn = local_db.get_connection()
    now = _now()
    try:
        cur = conn.execute(
            "UPDATE kg_sleep_queue SET processed_at = ? WHERE session_id = ?",
            (now, session_id),
        )
        conn.commit()
        if cur.rowcount == 0:
            log.warning("mark_processed: no kg_sleep_queue row for session %s.", session_id)
            return False
        log.debug("mark_processed: session %s marked done.", session_id)
        return True
    except Exception as e:
        conn.rollback()
        log.error("mark_processed(%s) error: %s", session_id, e, exc_info=True)
        return False


def count_pending() -> int:
    """
    Number of sessions still awaiting sleep processing. Used for
    progress reporting (e.g. "3 sessions pending in sleep queue").
    Returns -1 on error.
    """
    conn = local_db.get_connection()
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM kg_sleep_queue WHERE processed_at IS NULL"
        ).fetchone()[0]
    except Exception as e:
        log.error("count_pending error: %s", e, exc_info=True)
        return -1


def delete_processed(older_than_days: int = 7) -> int:
    """
    Delete processed rows older than `older_than_days`. Pending rows
    (processed_at IS NULL) are never touched by this function regardless
    of age — only cleanup of already-completed work.

    Returns the number of rows deleted, or -1 on error.
    """
    conn = local_db.get_connection()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=older_than_days)).isoformat()
    try:
        cur = conn.execute(
            """
            DELETE FROM kg_sleep_queue
            WHERE processed_at IS NOT NULL AND processed_at < ?
            """,
            (cutoff,),
        )
        conn.commit()
        deleted = cur.rowcount
        if deleted:
            log.info("delete_processed: removed %d processed kg_sleep_queue row(s) older than %d days.",
                      deleted, older_than_days)
        return deleted
    except Exception as e:
        conn.rollback()
        log.error("delete_processed error: %s", e, exc_info=True)
        return -1