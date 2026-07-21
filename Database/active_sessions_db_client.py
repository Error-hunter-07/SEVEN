"""
Database/active_sessions_db_client.py

CRUD for the active_sessions table (Database/local_db.py). This table
exists purely as a crash-durability marker for episodic memory:

  - start_session() is called once, at on_session_start.
  - heartbeat() is called after every successful turn (cheap single-row
    UPDATE) — SQLite's WAL mode fsyncs committed writes, so this
    survives a hard crash (kill -9, power loss), unlike anything kept
    only in the in-process history_manager.messages list.
  - close_session() is called once a session ends cleanly (session_lifecycle
    on_session_end has already written the real episodic_memory row by
    that point) — the marker row is deleted since it's no longer needed.

Anything still 'in_progress' the NEXT time a process starts up belongs
to a session that never got a clean shutdown — see
SessionManager/session_lifecycle.py's crash-recovery sweep.
"""

from datetime import datetime, timezone

import Database.local_db as local_db
from GlobalHelpers.logger import get_logger

log = get_logger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def start_session(session_id):
    """
    Idempotent-ish: if a row already exists for this session_id (shouldn't
    normally happen — session ids are freshly generated per process), it's
    left alone rather than clobbered, since the process may be recovering
    from a partial start of the same session.
    """
    conn = local_db.get_connection()
    now = _now()
    try:
        conn.execute(
            """
            INSERT INTO active_sessions (session_id, started_at, last_turn_at, turn_count, status)
            VALUES (?, ?, ?, 0, 'in_progress')
            ON CONFLICT(session_id) DO NOTHING
            """,
            (session_id, now, now),
        )
        conn.commit()
        return True
    except Exception as e:
        log.error("start_session error: %s: %s", type(e).__name__, e, exc_info=True)
        conn.rollback()
        return False


def heartbeat(session_id):
    """Bumps turn_count by 1 and refreshes last_turn_at. Called after
    every successful assistant turn."""
    conn = local_db.get_connection()
    now = _now()
    try:
        cur = conn.execute(
            """
            UPDATE active_sessions
            SET turn_count = turn_count + 1, last_turn_at = ?
            WHERE session_id = ?
            """,
            (now, session_id),
        )
        if cur.rowcount == 0:
            # No marker row yet (e.g. heartbeat raced session start) —
            # self-heal by creating one rather than silently losing the turn.
            log.warning("heartbeat: no active_sessions row for %s — creating one now.", session_id)
            start_session(session_id)
            conn.execute(
                "UPDATE active_sessions SET turn_count = 1, last_turn_at = ? WHERE session_id = ?",
                (now, session_id),
            )
        conn.commit()
        return True
    except Exception as e:
        log.error("heartbeat error: %s", e, exc_info=True)
        conn.rollback()
        return False


def get_turn_count(session_id):
    conn = local_db.get_connection()
    try:
        cur = conn.execute(
            "SELECT turn_count FROM active_sessions WHERE session_id = ?",
            (session_id,),
        )
        row = cur.fetchone()
        return row["turn_count"] if row else 0
    except Exception as e:
        log.error("get_turn_count error: %s", e, exc_info=True)
        return 0


def get_started_at(session_id):
    conn = local_db.get_connection()
    try:
        cur = conn.execute(
            "SELECT started_at FROM active_sessions WHERE session_id = ?",
            (session_id,),
        )
        row = cur.fetchone()
        return row["started_at"] if row else None
    except Exception as e:
        log.error("get_started_at error: %s", e, exc_info=True)
        return None


def is_session_active(session_id):
    conn = local_db.get_connection()
    try:
        cur = conn.execute(
            "SELECT status FROM active_sessions WHERE session_id = ?",
            (session_id,),
        )
        row = cur.fetchone()
        return bool(row) and row["status"] == "in_progress"
    except Exception as e:
        log.error("is_session_active error: %s", e, exc_info=True)
        # Fail open: if we can't tell, assume active so on_session_end
        # doesn't skip a legitimate close.
        return True


def close_session(session_id):
    """Called once a session's episodic_memory row has been written
    successfully. Deletes the marker — a clean close means there's
    nothing left to recover."""
    conn = local_db.get_connection()
    try:
        conn.execute("DELETE FROM active_sessions WHERE session_id = ?", (session_id,))
        conn.commit()
        return True
    except Exception as e:
        log.error("close_session error: %s", e, exc_info=True)
        conn.rollback()
        return False


def get_stale_sessions(exclude_session_id):
    """All sessions still marked 'in_progress' that are NOT the caller's
    own current session — i.e. leftovers from a previous process that
    never shut down cleanly."""
    conn = local_db.get_connection()
    try:
        cur = conn.execute(
            """
            SELECT session_id, started_at, last_turn_at, turn_count, status
            FROM active_sessions
            WHERE status = 'in_progress' AND session_id != ?
            """,
            (exclude_session_id,),
        )
        return [dict(r) for r in cur.fetchall()]
    except Exception as e:
        log.error("get_stale_sessions error: %s", e, exc_info=True)
        return []
