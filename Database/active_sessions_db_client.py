"""
Database/active_sessions_db_client.py

CRUD for the active_sessions table. This table is both a crash-durability
marker AND the live scratch space for whatever the current session needs
to survive a crash:

  - start_session()              called once, at on_session_start
  - heartbeat()                  called after every successful turn
  - append_chunk_summary()       called every 5 turns by chunk_summary_worker
  - save_full_conversation()     called every turn (cheap overwrite) as a
                                  last-resort backup for the first few
                                  turns before any chunk summary exists
  - append_semantic_memory_id()  called by session_memory_tracker every
                                  time a NEW semantic memory is created
  - close_session()              called once a session ends cleanly —
                                  deletes the marker row entirely

Anything still 'in_progress' at the NEXT process's startup gets
recovered via SessionManager/session_lifecycle.py's crash-recovery
sweep, using whichever of chunk_summaries / full_conversation /
related_semantic_memory_ids survived.
"""

import json
from datetime import datetime, timezone

import Database.local_db as local_db
from GlobalHelpers.logger import get_logger

log = get_logger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def start_session(session_id):
    """Idempotent-ish: if a row already exists for this session_id
    (shouldn't normally happen), it's left alone rather than clobbered."""
    conn = local_db.get_connection()
    now = _now()
    try:
        conn.execute(
            """
            INSERT INTO active_sessions (
                session_id, started_at, last_turn_at, turn_count, status,
                chunk_summaries, full_conversation, related_semantic_memory_ids
            )
            VALUES (?, ?, ?, 0, 'in_progress', ?, ?, ?)
            ON CONFLICT(session_id) DO NOTHING
            """,
            (session_id, now, now, json.dumps([]), json.dumps([]), json.dumps([])),
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


def append_chunk_summary(session_id, summary_text):
    """Appends one rolling chunk summary. Read-modify-write under one
    connection — fine at this write frequency (once every 5 turns, not
    every turn)."""
    if not summary_text or not summary_text.strip():
        return False
    conn = local_db.get_connection()
    try:
        cur = conn.execute(
            "SELECT chunk_summaries FROM active_sessions WHERE session_id = ?",
            (session_id,),
        )
        row = cur.fetchone()
        if row is None:
            log.warning("append_chunk_summary: no active_sessions row for %s.", session_id)
            return False
        existing = json.loads(row["chunk_summaries"]) if row["chunk_summaries"] else []
        existing.append(summary_text.strip())
        conn.execute(
            "UPDATE active_sessions SET chunk_summaries = ? WHERE session_id = ?",
            (json.dumps(existing), session_id),
        )
        conn.commit()
        return True
    except Exception as e:
        log.error("append_chunk_summary error: %s", e, exc_info=True)
        conn.rollback()
        return False


def get_chunk_summaries(session_id) -> list:
    conn = local_db.get_connection()
    try:
        cur = conn.execute(
            "SELECT chunk_summaries FROM active_sessions WHERE session_id = ?",
            (session_id,),
        )
        row = cur.fetchone()
        if row is None or not row["chunk_summaries"]:
            return []
        return json.loads(row["chunk_summaries"])
    except Exception as e:
        log.error("get_chunk_summaries error: %s", e, exc_info=True)
        return []


def save_full_conversation(session_id, messages: list) -> bool:
    """Overwrites the full conversation snapshot — cheap at conversation
    sizes this app deals with (thousands of tokens, not gigabytes), and
    it's a full overwrite rather than an append because SQLite has no
    cheap partial-text-append primitive. Called every turn as a
    last-resort crash backup for whatever hasn't been chunk-summarized
    yet."""
    conn = local_db.get_connection()
    try:
        conn.execute(
            "UPDATE active_sessions SET full_conversation = ? WHERE session_id = ?",
            (json.dumps(messages, default=str), session_id),
        )
        conn.commit()
        return True
    except Exception as e:
        log.error("save_full_conversation error: %s", e, exc_info=True)
        conn.rollback()
        return False


def get_full_conversation(session_id) -> list:
    conn = local_db.get_connection()
    try:
        cur = conn.execute(
            "SELECT full_conversation FROM active_sessions WHERE session_id = ?",
            (session_id,),
        )
        row = cur.fetchone()
        if row is None or not row["full_conversation"]:
            return []
        return json.loads(row["full_conversation"])
    except Exception as e:
        log.error("get_full_conversation error: %s", e, exc_info=True)
        return []


def append_semantic_memory_id(session_id, mem_id) -> bool:
    if not mem_id:
        return False
    conn = local_db.get_connection()
    try:
        cur = conn.execute(
            "SELECT related_semantic_memory_ids FROM active_sessions WHERE session_id = ?",
            (session_id,),
        )
        row = cur.fetchone()
        if row is None:
            # No active session row (e.g. called from a standalone script) — skip silently.
            return False
        existing = json.loads(row["related_semantic_memory_ids"]) if row["related_semantic_memory_ids"] else []
        if mem_id not in existing:
            existing.append(mem_id)
        conn.execute(
            "UPDATE active_sessions SET related_semantic_memory_ids = ? WHERE session_id = ?",
            (json.dumps(existing), session_id),
        )
        conn.commit()
        return True
    except Exception as e:
        log.error("append_semantic_memory_id error: %s", e, exc_info=True)
        conn.rollback()
        return False


def get_related_semantic_memory_ids(session_id) -> list:
    conn = local_db.get_connection()
    try:
        cur = conn.execute(
            "SELECT related_semantic_memory_ids FROM active_sessions WHERE session_id = ?",
            (session_id,),
        )
        row = cur.fetchone()
        if row is None or not row["related_semantic_memory_ids"]:
            return []
        return json.loads(row["related_semantic_memory_ids"])
    except Exception as e:
        log.error("get_related_semantic_memory_ids error: %s", e, exc_info=True)
        return []


def get_turn_count(session_id):
    conn = local_db.get_connection()
    try:
        cur = conn.execute("SELECT turn_count FROM active_sessions WHERE session_id = ?", (session_id,))
        row = cur.fetchone()
        return row["turn_count"] if row else 0
    except Exception as e:
        log.error("get_turn_count error: %s", e, exc_info=True)
        return 0


def get_started_at(session_id):
    conn = local_db.get_connection()
    try:
        cur = conn.execute("SELECT started_at FROM active_sessions WHERE session_id = ?", (session_id,))
        row = cur.fetchone()
        return row["started_at"] if row else None
    except Exception as e:
        log.error("get_started_at error: %s", e, exc_info=True)
        return None


def is_session_active(session_id):
    conn = local_db.get_connection()
    try:
        cur = conn.execute("SELECT status FROM active_sessions WHERE session_id = ?", (session_id,))
        row = cur.fetchone()
        return bool(row) and row["status"] == "in_progress"
    except Exception as e:
        log.error("is_session_active error: %s", e, exc_info=True)
        # Fail open: if we can't tell, assume active so on_session_end doesn't skip a legitimate close.
        return True


def close_session(session_id):
    """Called once a session's episodic memory has been written
    successfully. Deletes the entire row — a clean close means there's
    nothing left to recover, including the chunk summaries and
    conversation backup, which have already served their purpose."""
    conn = local_db.get_connection()
    try:
        conn.execute("DELETE FROM active_sessions WHERE session_id = ?", (session_id,))
        conn.commit()
        return True
    except Exception as e:
        log.error("close_session error: %s", e, exc_info=True)
        conn.rollback()
        return False


def get_stale_sessions(exclude_session_id) -> list:
    """All sessions still 'in_progress' that are NOT the caller's own
    current session — leftovers from a previous process that never shut
    down cleanly. Returns full rows (including chunk_summaries,
    full_conversation, related_semantic_memory_ids) so the crash-recovery
    path has everything it needs without a second query."""
    conn = local_db.get_connection()
    try:
        cur = conn.execute(
            """
            SELECT session_id, started_at, last_turn_at, turn_count, status,
                   chunk_summaries, full_conversation, related_semantic_memory_ids
            FROM active_sessions
            WHERE status = 'in_progress' AND session_id != ?
            """,
            (exclude_session_id,),
        )
        rows = []
        for r in cur.fetchall():
            d = dict(r)
            d["chunk_summaries"] = json.loads(d["chunk_summaries"]) if d.get("chunk_summaries") else []
            d["full_conversation"] = json.loads(d["full_conversation"]) if d.get("full_conversation") else []
            d["related_semantic_memory_ids"] = (
                json.loads(d["related_semantic_memory_ids"]) if d.get("related_semantic_memory_ids") else []
            )
            rows.append(d)
        return rows
    except Exception as e:
        log.error("get_stale_sessions error: %s", e, exc_info=True)
        return []