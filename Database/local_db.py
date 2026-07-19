"""
Database/local_db.py

Single embedded SQLite database for the app's local storage — starting
with working_memory. episodic_memory will be added as a second table in
this same file/module when that migration happens, so the whole app
ends up with exactly two embedded stores: this SQLite file for
structured/relational data, and ChromaDB's own persist_dir for semantic
memory's vector store. Neither requires an external server, which is
what makes packaging into a standalone executable practical.

Replaces Database/db.py (Postgres connection pool) for working memory.
Database/db.py itself is left untouched in case anything else still
references it — nothing currently does except the old
working_memory_db_client.py, which this file replaces.
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


def _ensure_schema() -> None:
    global _schema_ready
    if _schema_ready:
        return
    with _init_lock:
        if _schema_ready:
            return
        conn = _create_connection()
        try:
            conn.executescript("""
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

                -- episodic_memory table will be added here when that
                -- migration happens, in this same file, same DB_PATH.
            """)
            conn.commit()
            log.info("Local SQLite schema ready at %s", DB_PATH)
        finally:
            conn.close()
        _schema_ready = True


# Schema is created/verified as soon as this module is imported, so
# every caller can assume the table already exists.
_ensure_schema()