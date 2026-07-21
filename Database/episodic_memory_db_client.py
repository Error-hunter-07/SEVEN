"""
Database/episodic_memory_db_client.py

CRUD for the episodic_memory table (Database/local_db.py, same DB file
as working_memory). Mirrors working_memory_db_client.py's shape: a
_now() helper, a row-shaping helper, try/except/rollback around every
write, and JSON-encode-in / JSON-decode-out for list fields.

Unlike working_memory, there is no soft delete here (no `active`
column, no TTL) — episodic rows are meant to be preserved. Rows only
ever leave this table via the decay-by-summarization lifecycle
(MemoryManagement/episodic_memory/memory_lifecycle.py), which deletes
ORIGINAL rows only after they've been merged into a new summary row —
see delete_episodes().

Rows are returned as dicts rather than positional tuples (unlike
working_memory_db_client.py) — there's no pre-existing caller contract
to preserve here since this table is new, and a dict is easier to
extend safely as columns get added later.
"""

import json
import uuid
from datetime import datetime, timezone

import Database.local_db as local_db
from GlobalHelpers.logger import get_logger

log = get_logger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_dict(row) -> dict:
    return {
        "id": row["id"],
        "session_id": row["session_id"],
        "title": row["title"],
        "summary": row["summary"],
        "key_topics": json.loads(row["key_topics"]) if row["key_topics"] else [],
        "start_time": row["start_time"],
        "end_time": row["end_time"],
        "turn_count": row["turn_count"],
        "outcome": row["outcome"],
        "related_semantic_memory_ids": (
            json.loads(row["related_semantic_memory_ids"]) if row["related_semantic_memory_ids"] else []
        ),
        "chroma_id": row["chroma_id"],
        "importance": row["importance"],
        "decay_count": row["decay_count"],
        "merged_from": json.loads(row["merged_from"]) if row["merged_from"] else None,
        "created_at": row["created_at"],
    }


_SELECT_COLUMNS = """
    id, session_id, title, summary, key_topics, start_time, end_time,
    turn_count, outcome, related_semantic_memory_ids, chroma_id,
    importance, decay_count, merged_from, created_at
"""


def insert_episodic_memory(
    session_id,
    title,
    summary,
    start_time,
    end_time,
    key_topics=None,
    turn_count=0,
    outcome=None,
    related_semantic_memory_ids=None,
    chroma_id=None,
    importance=0.5,
    decay_count=0,
    merged_from=None,
):
    """
    Returns the new episode id on success, None on failure.

    `summary` is NOT NULL at the schema level — callers (session_lifecycle,
    the decay lifecycle) are expected to always have a usable summary in
    hand before calling this, falling back to a heuristic string if the
    LLM-generated one fails, rather than passing empty text through.
    """
    conn = local_db.get_connection()
    new_id = str(uuid.uuid4())
    now = _now()
    try:
        conn.execute(
            f"""
            INSERT INTO episodic_memory (
                id, session_id, title, summary, key_topics, start_time, end_time,
                turn_count, outcome, related_semantic_memory_ids, chroma_id,
                importance, decay_count, merged_from, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_id, session_id, title, summary,
                json.dumps(key_topics) if key_topics is not None else None,
                start_time, end_time, turn_count, outcome,
                json.dumps(related_semantic_memory_ids) if related_semantic_memory_ids is not None else None,
                chroma_id, importance, decay_count,
                json.dumps(merged_from) if merged_from is not None else None,
                now,
            ),
        )
        conn.commit()
        return new_id
    except Exception as e:
        log.error("insert_episodic_memory error: %s: %s", type(e).__name__, e, exc_info=True)
        conn.rollback()
        return None


def get_episodic_memory_by_session(session_id):
    conn = local_db.get_connection()
    try:
        cur = conn.execute(
            f"""
            SELECT {_SELECT_COLUMNS}
            FROM episodic_memory
            WHERE session_id = ?
            ORDER BY created_at DESC
            """,
            (session_id,),
        )
        return [_row_to_dict(r) for r in cur.fetchall()]
    except Exception as e:
        log.error("get_episodic_memory_by_session error: %s", e, exc_info=True)
        return None


def get_recent_episodes(limit=10):
    """
    Most recent episodes, newest first. `limit` is intentionally an
    open parameter — the LLM can ask for however many episodes of
    history it wants via the episodic-memory tool built on top of this.
    """
    conn = local_db.get_connection()
    try:
        cur = conn.execute(
            f"""
            SELECT {_SELECT_COLUMNS}
            FROM episodic_memory
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [_row_to_dict(r) for r in cur.fetchall()]
    except Exception as e:
        log.error("get_recent_episodes error: %s", e, exc_info=True)
        return None


def get_recent_episodes_capped():
    """Fixed-cap convenience wrapper (5) for places that just want
    'recent context' without exposing the count as a caller-controlled
    parameter — e.g. seeding a new session."""
    return get_recent_episodes(limit=5)


def get_episodes_by_decay_count(decay_count, older_than_iso, limit=None):
    """
    Used by the decay lifecycle to find same-level candidates for
    merging: rows at exactly `decay_count` whose created_at is older
    than `older_than_iso`. Oldest first, so a batch always merges the
    longest-waiting rows.
    """
    conn = local_db.get_connection()
    try:
        query = f"""
            SELECT {_SELECT_COLUMNS}
            FROM episodic_memory
            WHERE decay_count = ? AND created_at < ?
            ORDER BY created_at ASC
        """
        params = [decay_count, older_than_iso]
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        cur = conn.execute(query, params)
        return [_row_to_dict(r) for r in cur.fetchall()]
    except Exception as e:
        log.error("get_episodes_by_decay_count error: %s", e, exc_info=True)
        return None


def delete_episodes(ids):
    """
    Hard delete — only ever called by the decay lifecycle immediately
    after the rows named in `ids` have been successfully merged into a
    new summary row. Never called as a general-purpose delete path.
    """
    if not ids:
        return True
    conn = local_db.get_connection()
    try:
        placeholders = ",".join("?" for _ in ids)
        conn.execute(f"DELETE FROM episodic_memory WHERE id IN ({placeholders})", ids)
        conn.commit()
        return True
    except Exception as e:
        log.error("delete_episodes error: %s", e, exc_info=True)
        conn.rollback()
        return False
