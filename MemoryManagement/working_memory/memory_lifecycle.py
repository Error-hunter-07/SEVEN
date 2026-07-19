"""
MemoryManagement/working_memory/memory_lifecycle.py

Deletes expired working_memory rows so the SQLite file doesn't grow
forever. Filtering expired rows out of reads (see
Database/working_memory_db_client.py) keeps them from being surfaced to
the LLM, but doesn't reclaim disk space — this is the part that
actually removes them.

Mirrors MemoryManagement/semantic_memory/memory_lifecycle.py's pattern
(same folder shape, same "run once at startup on a background thread"
approach), just for working memory's TTL-based expiry instead of
semantic memory's importance-based decay/prune. Can also be called from
a future "recharge" session as part of general maintenance.
"""

from datetime import datetime, timezone

import Database.local_db as local_db
from GlobalHelpers.logger import get_logger

log = get_logger(__name__)


def prune_expired_working_memory() -> int:
    conn = local_db.get_connection()
    now = datetime.now(timezone.utc).isoformat()
    try:
        cur = conn.execute(
            "DELETE FROM working_memory WHERE expires_at IS NOT NULL AND expires_at <= ?",
            (now,),
        )
        conn.commit()
        deleted = cur.rowcount
        if deleted:
            log.info("Working memory prune: removed %d expired row(s).", deleted)
        else:
            log.debug("Working memory prune: nothing to prune.")
        return deleted
    except Exception as e:
        log.error("prune_expired_working_memory error: %s", e, exc_info=True)
        conn.rollback()
        return 0


def start() -> None:
    """Call once at process startup, alongside the existing semantic
    memory lifecycle start (see llm_client.py's bootstrap block)."""
    prune_expired_working_memory()