"""
MemoryManagement/episodic_memory/memory_lifecycle.py

Decay-by-summarization for episodic memory. Structurally mirrors
MemoryManagement/semantic_memory/memory_lifecycle.py (background daemon
thread, run once per session start) but operates over SQL rows in
episodic_memory instead of a Chroma collection, and never deletes
knowledge outright — old rows get merged into a single new summary row
instead of being dropped.

Escalating age thresholds by decay level:
  decay_count 0 -> 1  after 6 months  (fresh episodes)
  decay_count 1 -> 2  after 12 months (once-summarized episodes)
  decay_count 2+ -> N+1 after 24 months (twice-or-more-summarized episodes)

The threshold escalates because a decay_count=1 row is already a
summary of up to 20 sessions — it's coarser and more durable
information than a single fresh episode, so it's given longer before
being folded again.

Batching: rows at the same decay_count level only get merged once 20
of them have aged past that level's threshold (BATCH_SIZE). A level
with, say, 13 aged rows sits untouched until 7 more join it — this
avoids merging tiny, low-value batches.
"""

import threading
from datetime import datetime, timedelta, timezone

import Database.episodic_memory_db_client as episodic_memory_db_client
import MemoryManagement.episodic_memory.summarizer as summarizer
from GlobalHelpers.logger import get_logger

log = get_logger(__name__)

# ── tuneable constants ────────────────────────────────────────────────────────
BATCH_SIZE = 20
DECAY_THRESHOLDS_DAYS = {
    0: 180,   # 6 months  — fresh episodes -> first summary
    1: 365,   # 12 months — once-summarized -> second summary
}
DEFAULT_THRESHOLD_DAYS = 730  # 24 months — decay_count >= 2, and beyond
MAX_DECAY_LEVEL_SCANNED = 6   # sanity cap on how many levels we check per pass
# ─────────────────────────────────────────────────────────────────────────────


def _threshold_for(decay_count: int) -> int:
    return DECAY_THRESHOLDS_DAYS.get(decay_count, DEFAULT_THRESHOLD_DAYS)


def start() -> None:
    """Entry point. Call once at process startup (mirrors
    working_memory_lifecycle.start() / semantic memory_lifecycle.start()).
    Spawns a daemon thread so it never blocks the main loop."""
    t = threading.Thread(target=_run, daemon=True)
    t.name = "EpisodicMemoryLifecycle"
    t.start()


def _run() -> None:
    try:
        _decay_pass()
    except Exception:
        log.exception("Episodic memory lifecycle pass failed (non-fatal)")


def _decay_pass() -> None:
    now = datetime.now(timezone.utc)

    for level in range(0, MAX_DECAY_LEVEL_SCANNED):
        threshold_days = _threshold_for(level)
        cutoff_iso = (now - timedelta(days=threshold_days)).isoformat()

        candidates = episodic_memory_db_client.get_episodes_by_decay_count(
            decay_count=level, older_than_iso=cutoff_iso
        )
        if not candidates:
            continue

        merged_any = False
        while len(candidates) >= BATCH_SIZE:
            batch, candidates = candidates[:BATCH_SIZE], candidates[BATCH_SIZE:]
            if _merge_batch(batch, new_decay_count=level + 1):
                merged_any = True

        if merged_any:
            log.info("Episodic decay pass: merged batch(es) at decay_count=%d.", level)
        else:
            log.debug(
                "Episodic decay pass: %d aged row(s) at decay_count=%d, below batch size %d — waiting.",
                len(candidates), level, BATCH_SIZE,
            )


def _merge_batch(batch: list[dict], new_decay_count: int) -> bool:
    try:
        summary_result = summarizer.summarize_merge(batch)

        merged_ids = [ep["id"] for ep in batch]
        start_time = min(ep["start_time"] for ep in batch if ep.get("start_time"))
        end_time = max(ep["end_time"] for ep in batch if ep.get("end_time"))
        turn_count = sum(int(ep.get("turn_count") or 0) for ep in batch)
        importance = max(float(ep.get("importance") or 0.5) for ep in batch)
        representative_session_id = batch[0]["session_id"]

        related_ids: list[str] = []
        seen = set()
        for ep in batch:
            for mem_id in (ep.get("related_semantic_memory_ids") or []):
                if mem_id not in seen:
                    seen.add(mem_id)
                    related_ids.append(mem_id)

        new_id = episodic_memory_db_client.insert_episodic_memory(
            session_id=representative_session_id,
            title=summary_result["title"],
            summary=summary_result["summary"],
            key_topics=summary_result.get("key_topics", []),
            start_time=start_time,
            end_time=end_time,
            turn_count=turn_count,
            outcome=None,
            related_semantic_memory_ids=related_ids,
            importance=importance,
            decay_count=new_decay_count,
            merged_from=merged_ids,
        )

        if new_id is None:
            log.error("Episodic decay: failed to insert merged summary row — leaving source rows intact.")
            return False

        if not episodic_memory_db_client.delete_episodes(merged_ids):
            log.error(
                "Episodic decay: merged row %s inserted but failed to delete %d source rows — "
                "they'll be picked up again next pass (may cause a duplicate merge; non-fatal).",
                new_id, len(merged_ids),
            )
            return False

        return True
    except Exception:
        log.exception("Episodic decay: _merge_batch failed (non-fatal, source rows left intact).")
        return False
