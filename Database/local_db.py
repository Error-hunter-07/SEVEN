"""
Database/local_db.py

Single embedded SQLite database for the app's local storage. Holds three
tables now: working_memory, episodic_memory, and active_sessions — one
SQLite file for all structured/relational data, plus ChromaDB's own
persist_dir for semantic memory's vector store. Neither requires an
external server, which is what makes packaging into a standalone
executable practical.

Replaces Database/db.py (Postgres connection pool) for working memory.
Database/db.py itself is left untouched in case anything else still
references it — nothing currently does except the old
working_memory_db_client.py, which this file replaces.

SCHEMA INIT: each table has its own independent _ensure_*_schema()
function instead of one shared executescript() blob. This matters
because these tables are added at different times and owned by
different client modules — bundling them into a single script means a
DDL problem in one table's block could take a working, unrelated
table's init down with it, and every future schema tweak to any one
table would re-run the whole combined script. Keeping them independent
means each table's init succeeds or fails on its own.
"""

import sqlite3
import os
import threading
from GlobalHelpers.logger import get_logger

log = get_logger(__name__)

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "seven_local.db")

# One connection per thread. SQLite connections aren't safe to share
# across threads without care — rather than build a connection pool
# (which existed for Postgres to limit concurrent server connections,
# a concern that doesn't apply to an embedded file), we keep one
# connection per thread and let SQLite's own WAL-mode locking handle
# concurrency between them on the single file.
_local = threading.local()
_init_lock = threading.Lock()
_schema_ready = False


def _create_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")   # concurrent readers + one writer, no external lock needed
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.row_factory = sqlite3.Row              # dict-like column access by name
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


def _ensure_episodic_memory_schema(conn: sqlite3.Connection) -> None:
    """
    No `active`/soft-delete column here on purpose — episodic rows are
    meant to be preserved, not expired. The lifecycle for this table is
    decay-by-summarization (see MemoryManagement/episodic_memory/
    memory_lifecycle.py), not deletion: old rows get merged into a
    single LLM-generated summary row and only the ORIGINAL rows are
    removed, never the resulting knowledge itself.

    decay_count / merged_from support that: a fresh episode starts at
    decay_count=0; once enough same-level rows age past their
    threshold, they're merged into one new row at decay_count+1, and
    merged_from records which source episode ids fed into it.
    """
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS episodic_memory (
            id                           TEXT PRIMARY KEY,
            session_id                   TEXT NOT NULL,
            title                        TEXT,
            summary                      TEXT NOT NULL,
            key_topics                   TEXT,          -- JSON list
            start_time                   TEXT NOT NULL,
            end_time                     TEXT NOT NULL,
            turn_count                   INTEGER DEFAULT 0,
            outcome                      TEXT,          -- 'completed'|'abandoned'|'ongoing'|'interrupted'|NULL
            related_semantic_memory_ids  TEXT,          -- JSON list of ChromaDB ids
            chroma_id                    TEXT,          -- nullable, populated once embedding search over episodes exists
            importance                   REAL DEFAULT 0.5,
            decay_count                  INTEGER DEFAULT 0,
            merged_from                  TEXT,          -- JSON list of source episode ids, NULL until decayed at least once
            created_at                   TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_episodic_memory_session
            ON episodic_memory (session_id);

        CREATE INDEX IF NOT EXISTS idx_episodic_memory_decay_count
            ON episodic_memory (decay_count);
        """
    )


def _ensure_active_sessions_schema(conn: sqlite3.Connection) -> None:
    """
    Crash-durability marker table. Written at session start, updated on
    every turn (heartbeat), removed on a clean session end. Any row
    still status='in_progress' at the NEXT process's startup belongs to
    a session that never got a clean shutdown (crash, kill -9, power
    loss) — SessionManager/session_lifecycle.py sweeps this table at
    on_session_start to recover those as 'interrupted' episodes instead
    of losing that session's history entirely.
    """
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS active_sessions (
            session_id    TEXT PRIMARY KEY,
            started_at    TEXT NOT NULL,
            last_turn_at  TEXT,
            turn_count    INTEGER DEFAULT 0,
            status        TEXT NOT NULL DEFAULT 'in_progress'
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
            # Each table's init is independent — a failure in one does not
            # prevent the others from being created.
            for name, fn in (
                ("working_memory", _ensure_working_memory_schema),
                ("episodic_memory", _ensure_episodic_memory_schema),
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


# Schema is created/verified as soon as this module is imported, so
# every caller can assume the table already exists.
_ensure_schema()