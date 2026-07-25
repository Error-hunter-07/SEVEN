"""
Database/local_db.py

Single embedded SQLite database for the app's local structured storage.
Holds two tables now: working_memory and active_sessions.

CHANGED: episodic_memory is no longer a SQLite table. It moved to its
own ChromaDB collection (MemoryManagement/episodic_memory/episodic_memory_store.py)
because its primary access pattern is semantic search ("what did we
discuss about X"), not exact-match lookup — the same reasoning that
keeps working_memory and active_sessions in SQLite (always looked up by
session_id/id, never by meaning) but puts semantic_memory in Chroma.
Running a parallel SQLite copy of the same data alongside the Chroma
collection would have been pure redundancy, not defense in depth.

CHANGED: active_sessions gained three columns to make crash recovery
genuinely useful instead of a last-resort guess:
  - chunk_summaries: rolling 5-turn summaries, written live during the
    session (see LLMEngine/chunk_summary_worker.py), so a crash mid-way
    through a long session still has real narrative content to recover
    from, not just whatever's in working_memory.
  - full_conversation: the raw message history, overwritten each turn,
    as the last-resort fallback if even chunk summarization hasn't
    caught up yet (e.g. a crash in the first 4 turns).
  - related_semantic_memory_ids: which semantic-memory facts were
    created during this session, persisted turn-by-turn instead of only
    living in an in-memory dict (SessionManager/session_memory_tracker.py)
    that a crash would simply lose.

SCHEMA INIT: each table has its own independent _ensure_*_schema()
function rather than one shared executescript() blob, so a DDL problem
in one table's block can't take a working, unrelated table's init down
with it.
"""

import sqlite3
import os
import threading
from GlobalHelpers.logger import get_logger

log = get_logger(__name__)

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "seven_local.db")

_local = threading.local()
_init_lock = threading.Lock()
_schema_ready = False


def _create_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.row_factory = sqlite3.Row
    return conn


def get_connection() -> sqlite3.Connection:
    if getattr(_local, "conn", None) is None:
        _local.conn = _create_connection()
    return _local.conn


def _ensure_working_memory_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS working_memory (
            id            TEXT PRIMARY KEY,
            session_id    TEXT NOT NULL,
            memory_type   TEXT,
            key           TEXT,
            value         TEXT,              -- JSON-encoded
            priority      REAL DEFAULT 0.5,
            relevance     REAL DEFAULT 0.5,
            created_at    TEXT NOT NULL,
            updated_at    TEXT NOT NULL,
            expires_at    TEXT,
            source        TEXT,
            tags          TEXT,               -- JSON-encoded list
            access_count  INTEGER DEFAULT 0,
            last_accessed TEXT,
            active        INTEGER DEFAULT 1
        );

        CREATE INDEX IF NOT EXISTS idx_working_memory_session
            ON working_memory (session_id, active);
        """
    )


def _ensure_active_sessions_schema(conn: sqlite3.Connection) -> None:
    """
    Crash-durability marker + live scratch table for the current session.
    Written at session start, updated on every turn, cleared on a clean
    session end. Any row still status='in_progress' at the NEXT
    process's startup belongs to a session that never got a clean
    shutdown — see SessionManager/session_lifecycle.py's crash-recovery
    sweep, which now has real material (chunk_summaries,
    full_conversation, related_semantic_memory_ids) to recover from
    instead of just a working_memory snippet.
    """
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS active_sessions (
            session_id                    TEXT PRIMARY KEY,
            started_at                    TEXT NOT NULL,
            last_turn_at                  TEXT,
            turn_count                    INTEGER DEFAULT 0,
            status                        TEXT NOT NULL DEFAULT 'in_progress',
            chunk_summaries                TEXT,   -- JSON list of strings, appended every 5 turns
            full_conversation             TEXT,   -- JSON list of {role, content} messages, overwritten each turn
            related_semantic_memory_ids   TEXT     -- JSON list of ChromaDB ids created this session
        );
        """
    )


def _ensure_schema() -> None:
    global _schema_ready
    if _schema_ready:
        return
    with _init_lock:
        if _schema_ready:
            return
        conn = _create_connection()
        try:
            for name, fn in (
                ("working_memory", _ensure_working_memory_schema),
                ("active_sessions", _ensure_active_sessions_schema),
            ):
                try:
                    fn(conn)
                    conn.commit()
                    log.info("Local SQLite schema ready for '%s' at %s", name, DB_PATH)
                except Exception:
                    conn.rollback()
                    log.exception("Failed to initialize schema for '%s' (non-fatal, other tables continue).", name)
        finally:
            conn.close()
        _schema_ready = True


_ensure_schema()