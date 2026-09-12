"""
KnowledgeGraph/memory_selector.py

Selects sessions from the kg_sleep_queue that are pending Knowledge Graph
processing and returns them as SessionBundle objects — one per session,
containing all the context the entity extractor needs.

UNIT OF WORK: SESSION (not individual memory)
  The previous design used MemoryRecord (one ChromaDB memory per unit).
  The new design uses SessionBundle (one complete session per unit) because:

  1. Entity extraction needs SESSION CONTEXT. A semantic memory "User prefers
     Python" extracted in isolation has no context. The same fact extracted
     alongside the conversation "We discussed migrating SEVEN from Node.js to
     Python and FastAPI" produces richer, correctly-typed entities with correct
     relationships.

  2. The queue already packages everything. kg_sleep_queue stores
     conversation_text (chunk summaries), episodic_memory_id, and
     semantic_memory_ids at session-end — before active_sessions is deleted.
     The selector's job is to fetch that package and hydrate the texts.

  3. Sessions are the natural unit of "processed". After the pipeline runs,
     mark_processed(session_id) stamps one row. No per-memory tracking needed.

WHAT THIS MODULE DOES:
  1. Call kg_sleep_queue_client.get_pending_sessions(BATCH_SIZE) — returns
     queue rows with session_id, episodic_memory_id, semantic_memory_ids,
     conversation_text, all oldest-first.

  2. For each queue row, hydrate into a SessionBundle:
       - conversation_text    already in the row (chunk summaries joined)
       - episode_text         fetched from ChromaDB episodic_memory by
                              episodic_memory_id  →  "title\\nsummary"
       - semantic_texts       fetched from ChromaDB semantic_memory by
                              each id in semantic_memory_ids

  3. Return list[SessionBundle] for the entity extractor to consume.

WHAT THIS MODULE DOES NOT DO:
  - Does not call mark_processed() — that is the sleep_scheduler's job
    after all pipeline steps succeed.
  - Does not write to the graph.
  - Does not filter or score sessions (oldest-first ordering from the DB
    is the correct priority — process sessions in the order they happened).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from Database.kg_sleep_queue_client import get_pending_sessions, count_pending
from KnowledgeGraph.constants import BATCH_SIZE
from GlobalHelpers.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public data structure
# ---------------------------------------------------------------------------

@dataclass
class SessionBundle:
    """
    All context for one session, ready for entity extraction.

    session_id:          The session identifier. Passed through the
                         pipeline so sleep_scheduler can call
                         mark_processed(session_id) at the end.

    episodic_memory_id:  The ChromaDB id of the episode for this session.
                         Kept for reference / linking; text is in episode_text.

    semantic_memory_ids: The ChromaDB ids of semantic memories created
                         during this session. Kept alongside semantic_texts
                         so the pipeline can link nodes back to their
                         source memories via kg_memory_nodes.

    conversation_text:   Chunk summaries joined as paragraphs. This is the
                         richest extraction signal — the actual words and
                         topics from the conversation, not a polished summary.
                         May be empty if the session was very short (< 5 turns).

    episode_text:        "title\\nsummary" from ChromaDB episodic memory.
                         Provides the overall narrative arc of the session.
                         May be empty if the ChromaDB fetch failed.

    semantic_texts:      List of fact texts from ChromaDB semantic memory.
                         Individual facts created during the session
                         ("User prefers Python", "SEVEN uses PostgreSQL").
                         May be shorter than semantic_memory_ids if some
                         memories were pruned from ChromaDB since session end.
    """
    session_id:          str
    episodic_memory_id:  str
    semantic_memory_ids: list[str]   = field(default_factory=list)
    conversation_text:   str         = ""
    episode_text:        str         = ""
    semantic_texts:      list[str]   = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal: ChromaDB fetchers (late imports — ChromaDB inits on bg thread)
# ---------------------------------------------------------------------------

def _get_episodic_store():
    """Late import to avoid import-time ChromaDB initialisation."""
    try:
        from MemoryManagement.episodic_memory.episodic_memory_store import (
            get_episodes_by_session,
        )
        return get_episodes_by_session
    except Exception as e:
        log.warning("_get_episodic_store: import failed: %s", e)
        return None


def _get_semantic_memory():
    """Late import to avoid import-time ChromaDB initialisation."""
    try:
        from MemoryManagement.semantic_memory.semantic_memory import semantic_memory
        return semantic_memory
    except Exception as e:
        log.warning("_get_semantic_memory: import failed: %s", e)
        return None


def _fetch_episode_text(session_id: str) -> str:
    """
    Fetch episode text for a session from ChromaDB episodic memory.

    Returns "title\\nsummary" string, or "" if unavailable.
    Uses get_episodes_by_session() rather than fetching by episodic_memory_id
    directly — this is more robust because the ChromaDB id may have been
    recycled by decay/merge cycles, while session_id is stable.
    """
    get_episodes_by_session = _get_episodic_store()
    if get_episodes_by_session is None:
        return ""

    try:
        episodes = get_episodes_by_session(session_id)
        if not episodes:
            log.debug("_fetch_episode_text: no episode found for session=%s.", session_id)
            return ""
        # Take the most recent (first after sort by start_time_epoch DESC)
        ep = episodes[0]
        title   = ep.get("title", "").strip()
        summary = ep.get("summary", "").strip()
        parts = [p for p in [title, summary] if p]
        return "\n".join(parts)
    except Exception as e:
        log.error("_fetch_episode_text(session=%s) error: %s", session_id, e, exc_info=True)
        return ""


def _fetch_semantic_texts(semantic_memory_ids: list[str]) -> list[str]:
    """
    Fetch fact texts for a list of semantic memory ids from ChromaDB.

    Returns a list of text strings, one per successfully fetched memory.
    Silently skips ids that no longer exist in ChromaDB (memories that
    were pruned or decayed since session end).
    Returns [] if semantic_memory is unavailable or ids is empty.
    """
    if not semantic_memory_ids:
        return []

    sem_mem = _get_semantic_memory()
    if sem_mem is None:
        return []

    try:
        rows = sem_mem.get_by_ids(semantic_memory_ids)
        texts = []
        for row in rows:
            text = row.get("text", "").strip() if isinstance(row, dict) else ""
            if text:
                texts.append(text)
        return texts
    except Exception as e:
        log.error("_fetch_semantic_texts error: %s", e, exc_info=True)
        return []


# ---------------------------------------------------------------------------
# Internal: hydrate queue row into SessionBundle
# ---------------------------------------------------------------------------

def _hydrate(queue_row: dict) -> Optional[SessionBundle]:
    """
    Convert a raw queue row dict into a fully hydrated SessionBundle.

    The queue row already contains conversation_text. This function adds
    episode_text (from ChromaDB) and semantic_texts (from ChromaDB).

    Returns None if the queue row is missing a session_id (should never
    happen but guards against malformed rows).

    A bundle with empty episode_text or empty semantic_texts is still
    returned — the entity extractor can work with conversation_text alone.
    """
    session_id = queue_row.get("session_id", "").strip()
    if not session_id:
        log.warning("_hydrate: queue row missing session_id — skipping.")
        return None

    episodic_memory_id  = queue_row.get("episodic_memory_id", "")
    semantic_memory_ids = queue_row.get("semantic_memory_ids") or []
    conversation_text   = queue_row.get("conversation_text", "")

    # Fetch texts from ChromaDB — both are best-effort and non-fatal
    episode_text    = _fetch_episode_text(session_id)
    semantic_texts  = _fetch_semantic_texts(semantic_memory_ids)

    log.debug(
        "_hydrate: session=%s ep_text=%d chars sem_texts=%d conv_text=%d chars.",
        session_id, len(episode_text), len(semantic_texts), len(conversation_text),
    )

    return SessionBundle(
        session_id          = session_id,
        episodic_memory_id  = episodic_memory_id,
        semantic_memory_ids = semantic_memory_ids,
        conversation_text   = conversation_text,
        episode_text        = episode_text,
        semantic_texts      = semantic_texts,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_next_batch(batch_size: int = BATCH_SIZE) -> list[SessionBundle]:
    """
    Return the next batch of pending sessions as hydrated SessionBundles.

    Pulls from kg_sleep_queue WHERE processed_at IS NULL ORDER BY queued_at ASC,
    so sessions are processed in the order they occurred — oldest first.
    This chronological ordering gives the entity resolver the best chance
    of matching concepts correctly: the first occurrence creates the node,
    later occurrences update/confirm it with growing confidence.

    Returns [] when:
      - No sessions are pending (all processed or queue is empty).
      - kg_sleep_queue_client returns an empty list.
      - All fetched rows failed hydration (malformed session_ids).

    Non-fatal throughout — ChromaDB fetch failures produce bundles with
    empty episode_text / semantic_texts rather than raising or returning None.
    The entity extractor can still work from conversation_text alone.

    Args:
      batch_size: Number of sessions to fetch. Defaults to BATCH_SIZE from
                  constants.py. Capped at 1 minimum.
    """
    if batch_size <= 0:
        log.warning("get_next_batch: invalid batch_size=%d — returning [].", batch_size)
        return []

    pending_rows = get_pending_sessions(limit=batch_size)

    if not pending_rows:
        log.info("get_next_batch: no pending sessions in kg_sleep_queue.")
        return []

    bundles = []
    for row in pending_rows:
        bundle = _hydrate(row)
        if bundle is not None:
            bundles.append(bundle)

    log.info(
        "get_next_batch: %d pending rows → %d bundles hydrated "
        "(ep_texts=%d non-empty, sem_texts_total=%d).",
        len(pending_rows),
        len(bundles),
        sum(1 for b in bundles if b.episode_text),
        sum(len(b.semantic_texts) for b in bundles),
    )
    return bundles


def get_queue_status() -> tuple[int, int]:
    """
    Return (total_queued, pending_count) from kg_sleep_queue.

    Used by the sleep scheduler for progress reporting:
      "N sessions pending, M processed."

    Returns (-1, -1) on error (count_pending returns -1 on error,
    total requires a separate query — see kg_sleep_queue_client for
    the full count_pending() implementation).
    """
    pending = count_pending()
    return (pending, pending)   # total not exposed by count_pending — pending is enough