"""
MemoryManagement/episodic_memory/episodic_memory_store.py

CRUD for episodic memory, backed by its own ChromaDB collection
(Database/chroma_db.py's episodic_memory_db) instead of a SQLite table.
Replaces the earlier Database/episodic_memory_db_client.py entirely —
episodic memory's primary access pattern is semantic search ("what did
we discuss about X"), which SQLite has no native support for; keeping a
parallel SQLite copy of the same rows just for structured filtering
would have been pure redundancy, not defense in depth, since Chroma's
metadata `where` filters already cover every structured query this
table needs (decay_count, session_id, time cutoffs).

The embedded document text is `title + "\\n" + summary` — both matter
for search relevance, so both go into what gets embedded, not just one.

METADATA SCHEMA (kept intentionally richer than the current code paths
use, so future use cases — the graph-memory idea, procedural memory
cross-references — don't require a migration later):
    session_id                     str
    title                          str
    key_topics                     str   (JSON-encoded list)
    start_time_iso / end_time_iso  str   (human-readable)
    start_time_epoch / end_time_epoch  float  (for $lt/$gt filtering —
                                                Chroma's numeric operators
                                                need real numbers, not
                                                ISO strings)
    turn_count                     int
    outcome                        str   ('completed'|'abandoned'|'ongoing'|'interrupted'|'')
    related_semantic_memory_ids    str   (JSON-encoded list)
    entities_mentioned             str   (JSON-encoded list — RESERVED,
                                          not populated yet; kept for the
                                          future graph-memory work so
                                          past episodes don't need
                                          reprocessing to backfill it)
    importance                     float
    decay_count                    int
    merged_from                    str   (JSON-encoded list, '' until decayed once)
    access_count                   int   (RESERVED — only ever bumped by
                                          explicit tool-driven recall,
                                          never by the passive seed or
                                          the deterministic trigger; see
                                          working_memory's dead
                                          access_count column for why
                                          that discipline matters)
    last_accessed_epoch            float (RESERVED, same rule as above)
    created_at_epoch                float
"""

import json
import time
import uuid
from datetime import datetime, timezone

from GlobalHelpers.logger import get_logger

log = get_logger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso_to_epoch(iso_str: str) -> float:
    try:
        return datetime.fromisoformat(iso_str).timestamp()
    except Exception:
        return time.time()


def _metadata_from_fields(
    session_id, title, key_topics, start_time_iso, end_time_iso,
    turn_count, outcome, related_semantic_memory_ids,
    importance, decay_count, merged_from,
    entities_mentioned=None, access_count=0, last_accessed_epoch=None,
) -> dict:
    now_epoch = time.time()
    return {
        "session_id":                   session_id or "",
        "title":                        title or "",
        "key_topics":                   json.dumps(key_topics or []),
        "start_time_iso":               start_time_iso,
        "end_time_iso":                 end_time_iso,
        "start_time_epoch":             _iso_to_epoch(start_time_iso),
        "end_time_epoch":                _iso_to_epoch(end_time_iso),
        "turn_count":                   int(turn_count or 0),
        "outcome":                      outcome or "",
        "related_semantic_memory_ids":  json.dumps(related_semantic_memory_ids or []),
        "entities_mentioned":           json.dumps(entities_mentioned or []),
        "importance":                   float(importance if importance is not None else 0.5),
        "decay_count":                  int(decay_count or 0),
        "merged_from":                  json.dumps(merged_from) if merged_from else "",
        "access_count":                 int(access_count or 0),
        "last_accessed_epoch":          float(last_accessed_epoch) if last_accessed_epoch else now_epoch,
        "created_at_epoch":             now_epoch,
    }


def _row_from_result(result: dict) -> dict:
    """Normalizes a raw ChromaClient result ({id, text, metadata[, score]})
    into a flat dict with JSON fields decoded back into Python lists."""
    meta = result["metadata"]
    return {
        "id": result["id"],
        "text": result["text"],
        "score": result.get("score"),
        "session_id": meta.get("session_id", ""),
        "title": meta.get("title", ""),
        "summary": result["text"].split("\n", 1)[1] if "\n" in result["text"] else result["text"],
        "key_topics": json.loads(meta.get("key_topics") or "[]"),
        "start_time_iso": meta.get("start_time_iso"),
        "end_time_iso": meta.get("end_time_iso"),
        "start_time_epoch": meta.get("start_time_epoch"),
        "end_time_epoch": meta.get("end_time_epoch"),
        "turn_count": meta.get("turn_count", 0),
        "outcome": meta.get("outcome") or None,
        "related_semantic_memory_ids": json.loads(meta.get("related_semantic_memory_ids") or "[]"),
        "entities_mentioned": json.loads(meta.get("entities_mentioned") or "[]"),
        "importance": meta.get("importance", 0.5),
        "decay_count": meta.get("decay_count", 0),
        "merged_from": json.loads(meta["merged_from"]) if meta.get("merged_from") else None,
        "access_count": meta.get("access_count", 0),
        "last_accessed_epoch": meta.get("last_accessed_epoch"),
        "created_at_epoch": meta.get("created_at_epoch"),
    }


def _get_db():
    """Always read the live db reference at call time (not import time) —
    same stale-reference lesson as semantic_memory.py: ChromaDB
    initializes on a background thread, so a module-level import taken
    too early would freeze in on None."""
    import Database.chroma_db as chroma_module
    return chroma_module.episodic_memory_db


def insert_episode(
    session_id,
    title,
    summary,
    start_time_iso,
    end_time_iso,
    key_topics=None,
    turn_count=0,
    outcome=None,
    related_semantic_memory_ids=None,
    importance=0.5,
    decay_count=0,
    merged_from=None,
) -> str | None:
    """Embeds title+summary and stores the episode. Returns the new
    episode id on success, None on failure (including DB not ready)."""
    db = _get_db()
    if db is None:
        log.warning("insert_episode: episodic Chroma collection not available.")
        return None

    summary = (summary or "").strip()
    if not summary:
        log.warning("insert_episode: empty summary — refusing to store an unusable episode.")
        return None

    episode_id = f"ep_{uuid.uuid4().hex[:12]}"
    document_text = f"{title or ''}\n{summary}".strip()

    metadata = _metadata_from_fields(
        session_id=session_id, title=title, key_topics=key_topics,
        start_time_iso=start_time_iso, end_time_iso=end_time_iso,
        turn_count=turn_count, outcome=outcome,
        related_semantic_memory_ids=related_semantic_memory_ids,
        importance=importance, decay_count=decay_count, merged_from=merged_from,
    )

    success = db.add(id=episode_id, text=document_text, metadata=metadata)
    if not success:
        log.error("insert_episode: Chroma add() failed for session %s.", session_id)
        return None
    log.info("Episode stored: %s (session=%s, decay_count=%d)", episode_id, session_id, decay_count)
    return episode_id


def search_episodes(query: str, k: int = 5) -> list[dict]:
    """Semantic search over episode title+summary text."""
    db = _get_db()
    if db is None:
        return []
    results = db.search(query=query, k=k)
    return [_row_from_result(r) for r in results]


def get_recent_episodes(limit: int = 5) -> list[dict]:
    """
    Most recent episodes, newest first. Chroma's .get() has no ORDER BY,
    so this fetches a reasonably-bounded superset (limit * 4, capped)
    and sorts by start_time_epoch in Python — cheap at the scale this
    app operates at (dozens to low hundreds of episodes, not millions).
    """
    db = _get_db()
    if db is None:
        return []
    fetch_cap = min(max(limit * 4, 20), 200)
    raw = db.get_by_metadata(where={}, limit=fetch_cap) if hasattr(db, "get_by_metadata") else []
    rows = [_row_from_result(r) for r in raw]
    rows.sort(key=lambda r: r.get("start_time_epoch") or 0, reverse=True)
    return rows[:limit]


def get_recent_episodes_capped() -> list[dict]:
    """Fixed-cap convenience wrapper (2, per the agreed passive-seed
    budget) for on_session_start — keeps the token cost of the always-on
    context floor low."""
    return get_recent_episodes(limit=2)


def get_episodes_by_decay_count(decay_count: int, older_than_epoch: float, limit: int | None = None) -> list[dict]:
    """Used by the decay lifecycle to find same-level candidates for
    merging: rows at exactly `decay_count` whose start_time_epoch is
    older than the cutoff. Sorted oldest-first in Python so a batch
    always merges the longest-waiting rows."""
    db = _get_db()
    if db is None:
        return []
    where = {"$and": [{"decay_count": decay_count}, {"start_time_epoch": {"$lt": older_than_epoch}}]}
    raw = db.get_by_metadata(where=where) if hasattr(db, "get_by_metadata") else []
    rows = [_row_from_result(r) for r in raw]
    rows.sort(key=lambda r: r.get("start_time_epoch") or 0)
    if limit is not None:
        rows = rows[:limit]
    return rows


def delete_episodes(ids: list[str]) -> bool:
    """Hard delete — only ever called by the decay lifecycle immediately
    after the rows named in `ids` have been successfully merged into a
    new summary row."""
    db = _get_db()
    if db is None:
        return False
    if not ids:
        return True
    if hasattr(db, "delete_many"):
        return db.delete_many(ids)
    # Fallback for a backend without batch delete
    ok = True
    for episode_id in ids:
        ok = db.delete(episode_id) and ok
    return ok


def mark_recalled(episode_id: str) -> None:
    """Bumps access_count/last_accessed_epoch — call ONLY from explicit
    tool-driven recall (search_episodic_memory, the deterministic
    trigger), never from the passive 2-session seed. See the module
    docstring's note on why write-on-read is deliberately avoided
    elsewhere in this codebase (working_memory's dead access_count
    column, semantic_memory's retrieve() update_access fix)."""
    db = _get_db()
    if db is None:
        return
    existing = db.get(episode_id)
    if existing is None:
        return
    meta = existing["metadata"]
    db.update(
        id=episode_id,
        metadata={
            **meta,
            "access_count": int(meta.get("access_count", 0)) + 1,
            "last_accessed_epoch": time.time(),
        },
    )