"""
Database/working_memory_db_client.py

MIGRATED: was Postgres, now embedded SQLite (Database/local_db.py).

ADDED: TTL-based expiry to prevent unbounded growth. SQLite has no
time-based triggers (CREATE TRIGGER only fires on row events, not on a
clock), so expiry is enforced at the application level, in three parts:

  1. Every insert sets expires_at = now + WORKING_MEMORY_TTL_DAYS unless
     the caller explicitly provides one.
  2. Every update to a row REFRESHES expires_at — if you're still
     touching it, it's still relevant, so its clock resets (unless the
     caller explicitly overrides expires_at).
  3. Every read query filters out rows past their expires_at, so an
     expired memory never resurfaces even before cleanup has run.

Actual deletion of expired rows (so the DB file doesn't grow forever)
happens in Database/working_memory_lifecycle.py, run at startup —
mirroring the existing MemoryManagement/semantic_memory/memory_lifecycle.py
decay/prune pattern.

ADDED (reflection layer):
  get_by_type(session_id, memory_type)
      Session-scoped fetch by memory_type. Used by session-end
      consolidation to retrieve all reflection rows created during
      the current session before deciding which to keep or delete.

  get_active_reflections_all_sessions(limit)
      Cross-session fetch of all non-expired reflection rows, ordered
      by priority DESC then created_at DESC. Intentionally NOT filtered
      by session_id — reflections are meant to persist and influence
      behaviour across sessions. Used by PromptBuilder to build the
      live directive block that gets injected on every turn.

  delete_working_memory(memory_id)
      Hard DELETE by primary key. Used by session-end consolidation to
      permanently remove low-value reflection rows that the LLM judged
      as noise. Distinct from TTL expiry (which is passive/scheduled)
      — this is an immediate, intentional removal.
"""

import json
import uuid
from datetime import datetime, timedelta, timezone

import Database.local_db as local_db
from GlobalHelpers.logger import get_logger

log = get_logger(__name__)

WORKING_MEMORY_TTL_DAYS = 60  # ~2 months


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_expiry() -> str:
    return (datetime.now(timezone.utc) + timedelta(days=WORKING_MEMORY_TTL_DAYS)).isoformat()


def _row_to_tuple(row) -> tuple:
    """Preserves the same positional-tuple shape callers already expect:
    (id, memory_type, key, value, priority, relevance, created_at,
     updated_at, expires_at, source, tags)."""
    return (
        row["id"],
        row["memory_type"],
        row["key"],
        json.loads(row["value"]) if row["value"] is not None else None,
        row["priority"],
        row["relevance"],
        row["created_at"],
        row["updated_at"],
        row["expires_at"],
        row["source"],
        json.loads(row["tags"]) if row["tags"] else None,
    )


def insert_working_memory(session_id, memory_type, key, value, priority=0.5, relevance=0.5, source=None, tags=None, expires_at=None):
    conn = local_db.get_connection()
    new_id = str(uuid.uuid4())
    now = _now()
    if expires_at is None:
        expires_at = _default_expiry()
    try:
        conn.execute(
            """
            INSERT INTO working_memory (
                id, session_id, memory_type, key, value,
                priority, relevance, created_at, updated_at, expires_at,
                source, tags
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_id, session_id, memory_type, key, json.dumps(value),
                priority, relevance, now, now, expires_at,
                source, json.dumps(tags) if tags is not None else None,
            ),
        )
        conn.commit()
        return new_id
    except Exception as e:
        log.error("insert_working_memory error: %s: %s", type(e).__name__, e, exc_info=True)
        conn.rollback()
        return None


def get_working_memory(session_id):
    conn = local_db.get_connection()
    now = _now()
    try:
        cur = conn.execute(
            """
            SELECT id, memory_type, key, value, priority, relevance,
                   created_at, updated_at, expires_at, source, tags
            FROM working_memory
            WHERE session_id = ? AND active = 1
              AND (expires_at IS NULL OR expires_at > ?)
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (session_id, now),
        )
        return [_row_to_tuple(r) for r in cur.fetchall()]
    except Exception as e:
        log.error("get_working_memory error: %s", e, exc_info=True)
        return None


def update_working_memory(memory_id, key=None, value=None, priority=None, relevance=None, expires_at=None, source=None, tags=None):
    """
    Same failure-detection fix carried over from the Postgres version:
    refuses a no-op update when memory_id is missing, and checks
    cursor.rowcount to detect an update that matched zero rows.

    TTL: unless the caller explicitly passes expires_at, every update
    refreshes it to now + WORKING_MEMORY_TTL_DAYS — a row that's still
    being actively updated shouldn't decay just because it was created
    a while ago.
    """
    if not memory_id:
        log.warning("update_working_memory: called with no memory_id — refusing to run a no-op update.")
        return False

    conn = local_db.get_connection()
    try:
        fields = []
        values = []

        if key is not None:
            fields.append("key = ?")
            values.append(key)
        if value is not None:
            fields.append("value = ?")
            values.append(json.dumps(value))
        if priority is not None:
            fields.append("priority = ?")
            values.append(priority)
        if relevance is not None:
            fields.append("relevance = ?")
            values.append(relevance)
        if source is not None:
            fields.append("source = ?")
            values.append(source)
        if tags is not None:
            fields.append("tags = ?")
            values.append(json.dumps(tags))

        # TTL refresh: explicit expires_at wins; otherwise auto-renew
        # whenever the row is touched at all.
        if expires_at is not None:
            fields.append("expires_at = ?")
            values.append(expires_at)
        elif fields:
            fields.append("expires_at = ?")
            values.append(_default_expiry())

        if not fields:
            log.warning("update_working_memory: No fields to update.")
            return False

        fields.append("updated_at = ?")
        values.append(_now())
        values.append(memory_id)

        cur = conn.execute(
            f"UPDATE working_memory SET {', '.join(fields)} WHERE id = ?",
            values,
        )

        if cur.rowcount == 0:
            log.warning("update_working_memory: no row matched id=%s — nothing was updated.", memory_id)
            conn.rollback()
            return False

        conn.commit()
        return True
    except Exception as e:
        log.error("update_working_memory error: %s", e, exc_info=True)
        conn.rollback()
        return False


def get_all_current_session_working_memory(session_id):
    conn = local_db.get_connection()
    now = _now()
    try:
        cur = conn.execute(
            """
            SELECT id, memory_type, key, value, priority, relevance,
                   created_at, updated_at, expires_at, source, tags
            FROM working_memory
            WHERE session_id = ? AND active = 1
              AND (expires_at IS NULL OR expires_at > ?)
            ORDER BY created_at DESC
            """,
            (session_id, now),
        )
        return [_row_to_tuple(r) for r in cur.fetchall()]
    except Exception as e:
        log.error("get_all_current_session_working_memory error: %s", e, exc_info=True)
        return None


def get_by_type(session_id: str, memory_type: str) -> list[tuple]:
    """
    Returns all non-expired, active rows for a given session_id filtered
    by memory_type, ordered newest-first.

    Typical caller: session-end consolidation in
    SessionManager/session_lifecycle.py, which fetches all
    memory_type='reflection' rows written during the just-ended session
    before making the keep/delete/promote decision.

    Returns [] (not None) on an empty result so callers can iterate
    safely without a None-guard.
    """
    conn = local_db.get_connection()
    now = _now()
    try:
        cur = conn.execute(
            """
            SELECT id, memory_type, key, value, priority, relevance,
                   created_at, updated_at, expires_at, source, tags
            FROM working_memory
            WHERE session_id = ?
              AND memory_type = ?
              AND active = 1
              AND (expires_at IS NULL OR expires_at > ?)
            ORDER BY created_at DESC
            """,
            (session_id, memory_type, now),
        )
        return [_row_to_tuple(r) for r in cur.fetchall()]
    except Exception as e:
        log.error("get_by_type(%s, %s) error: %s", session_id, memory_type, e, exc_info=True)
        return []


def get_active_reflections_all_sessions(limit: int = 10) -> list[tuple]:
    """
    Returns the top `limit` non-expired reflection rows across ALL
    sessions, ordered by priority DESC then created_at DESC.

    No session_id filter — reflections are cross-session by design.
    A reflection written in session N should still influence behaviour
    in session N+5 until its expires_at passes or it is explicitly
    deleted by consolidation.

    Caller: PromptBuilder/prompt_builder.py — called on every turn to
    build the SELF-CORRECTION DIRECTIVES block. Must be fast (SQLite
    index scan on memory_type + active + expires_at). Limit keeps the
    directive block token-bounded; callers should not raise this above
    ~10 without also capping the formatted output by token count.
    """
    conn = local_db.get_connection()
    now = _now()
    try:
        cur = conn.execute(
            """
            SELECT id, memory_type, key, value, priority, relevance,
                   created_at, updated_at, expires_at, source, tags
            FROM working_memory
            WHERE memory_type = 'reflection'
              AND active = 1
              AND (expires_at IS NULL OR expires_at > ?)
            ORDER BY priority DESC, created_at DESC
            LIMIT ?
            """,
            (now, limit),
        )
        return [_row_to_tuple(r) for r in cur.fetchall()]
    except Exception as e:
        log.error("get_active_reflections_all_sessions error: %s", e, exc_info=True)
        return []


def delete_working_memory(memory_id: str) -> bool:
    """
    Hard DELETE of a single row by primary key.

    Used by session-end consolidation
    (SessionManager/session_lifecycle.py) to permanently remove
    reflection rows judged as low-value noise. This is an intentional,
    immediate removal — not the passive TTL expiry handled by
    MemoryManagement/working_memory/memory_lifecycle.py.

    Returns True if exactly one row was deleted, False if the id was
    not found or the delete failed.
    """
    if not memory_id:
        log.warning("delete_working_memory: called with empty memory_id — skipping.")
        return False
    conn = local_db.get_connection()
    try:
        cur = conn.execute(
            "DELETE FROM working_memory WHERE id = ?",
            (memory_id,),
        )
        conn.commit()
        if cur.rowcount == 0:
            log.warning(
                "delete_working_memory: no row matched id=%s — nothing deleted.",
                memory_id,
            )
            return False
        log.debug("delete_working_memory: deleted id=%s.", memory_id)
        return True
    except Exception as e:
        log.error("delete_working_memory(%s) error: %s", memory_id, e, exc_info=True)
        conn.rollback()
        return False