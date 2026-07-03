"""
memory_lifecycle.py
Handles importance decay and pruning for semantic memory.
Runs once per session start in a background daemon thread.

Decay model:
  - Memories older than DECAY_INTERVAL_DAYS get their importance multiplied
    by DECAY_FACTOR for each 30-day cycle elapsed.
  - Memories below MIN_IMPORTANCE are pruned unconditionally.
  - If collection exceeds MAX_MEMORIES, lowest-scored entries are pruned
    down to PRUNE_TARGET.

Score formula (for pruning priority):
  score = (importance * 0.7) + (capped_access_count * 0.3)
  Lower score = pruned first.
"""

import threading
from datetime import datetime, timezone
from GlobalHelpers.logger import get_logger

log = get_logger(__name__)

# ── tuneable constants ────────────────────────────────────────────────────────
MAX_MEMORIES        = 500    # hard cap before overflow pruning kicks in
PRUNE_TARGET        = 400    # prune down to this when cap is exceeded
DECAY_FACTOR        = 0.98   # importance multiplier per decay cycle
DECAY_INTERVAL_DAYS = 30     # one decay cycle = this many days
MIN_IMPORTANCE      = 0.15   # memories below this floor are always pruned
# ─────────────────────────────────────────────────────────────────────────────


def start(db) -> None:
    """
    Entry point. Call once from SemanticMemory.__init__().
    Spawns a daemon thread so it never blocks the main loop.

    Args:
        db: the ChromaClient singleton (semantic_memory_db)
    """
    t = threading.Thread(target=_run, args=(db,), daemon=True)
    t.name = "MemoryLifecycle"
    t.start()


# ── internal ──────────────────────────────────────────────────────────────────

def _run(db) -> None:
    try:
        _apply_decay(db)
        _prune(db)
    except Exception as e:
        log.exception("Memory lifecycle pass failed (non-fatal)")


def _apply_decay(db) -> None:
    """
    Decays importance of memories older than DECAY_INTERVAL_DAYS.
    Writes back only when the change is meaningful (> 0.001).
    """
    all_mem = db._collection.get(include=["metadatas", "documents"])
    if not all_mem["ids"]:
        log.debug("Decay pass: collection empty, nothing to do.")
        return

    now = datetime.now(timezone.utc)
    updates: list[tuple[str, float, dict]] = []

    for mem_id, meta in zip(all_mem["ids"], all_mem["metadatas"]):
        created_raw = meta.get("created_at", "")
        try:
            created = datetime.fromisoformat(created_raw).replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue  # malformed timestamp — skip silently

        age_days = (now - created).days
        if age_days < DECAY_INTERVAL_DAYS:
            continue  # too recent — no decay yet

        current = float(meta.get("importance", 0.5))
        cycles  = age_days // DECAY_INTERVAL_DAYS
        decayed = round(current * (DECAY_FACTOR ** cycles), 4)

        if abs(decayed - current) < 0.001:
            continue  # negligible change — skip the write

        updates.append((mem_id, decayed, meta))

    if not updates:
        log.debug("Decay pass: no memories old enough to decay.")
        return

    for mem_id, new_importance, meta in updates:
        db._collection.update(
            ids=[mem_id],
            metadatas=[{**meta, "importance": new_importance}]
        )

    log.info("Decayed %d memories.", len(updates))


def _prune(db) -> None:
    """
    Two-pass pruning:
      Pass 1 — delete anything below MIN_IMPORTANCE floor (unconditional).
      Pass 2 — if still over MAX_MEMORIES, delete lowest-scored until PRUNE_TARGET.
    """
    all_mem = db._collection.get(include=["metadatas"])
    if not all_mem["ids"]:
        return

    scored: list[tuple[float, str, float]] = []
    for mem_id, meta in zip(all_mem["ids"], all_mem["metadatas"]):
        importance   = float(meta.get("importance", 0.5))
        access_count = int(meta.get("access_count", 0))
        score = (importance * 0.7) + (min(access_count, 10) / 10 * 0.3)
        scored.append((score, mem_id, importance))

    # Pass 1: unconditional floor pruning
    to_delete = {mem_id for _, mem_id, imp in scored if imp < MIN_IMPORTANCE}

    # Pass 2: overflow pruning
    remaining = [(s, mid, imp) for s, mid, imp in scored if mid not in to_delete]
    if len(remaining) > MAX_MEMORIES:
        remaining.sort(key=lambda x: x[0])  # ascending — lowest score first
        overflow = len(remaining) - PRUNE_TARGET
        to_delete |= {mid for _, mid, _ in remaining[:overflow]}

    if not to_delete:
        log.debug("Prune pass: nothing to prune. Count: %d", db.count())
        return

    for mem_id in to_delete:
        db.delete(mem_id)

    log.info("Pruned %d memories. Remaining: %d", len(to_delete), db.count())