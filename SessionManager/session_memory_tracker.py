"""
SessionManager/session_memory_tracker.py

Small in-memory {session_id: [mem_ids]} map. semantic_memory.py's
store() calls record() every time it actually creates a NEW semantic
memory (not on a dedup hit against an existing one — see the note in
semantic_memory.py's store() for why dedup hits don't count). At
on_session_end, session_lifecycle.py calls get_and_clear(session_id) to
pull everything created during this session and drop it straight into
the new episode's related_semantic_memory_ids.

Keyed off GlobalHelpers.logger.get_session_id() rather than a value
passed around explicitly, because that contextvar already correctly
propagates into background threads (the batch-extraction worker) via
contextvars.copy_context() — see LLMEngine/extraction_worker.py. That
means this tracker works correctly for writes from any of the three
call sites that create semantic memory (background extraction, the
LLM's direct tool call, and session_lifecycle's own promotion writes),
without threading id-collection through all three individually.

Deliberately just a plain in-memory dict, not persisted — if the
process crashes before on_session_end runs, the crash-recovery path in
session_lifecycle.py rebuilds a best-effort episode from durable
working_memory rows instead, and related_semantic_memory_ids on that
recovered episode is simply left empty. That's an acceptable gap: the
semantic memories themselves are still safely in ChromaDB either way,
only the episode-to-facts link for that one interrupted session is
lost, not the facts.
"""

import threading

from GlobalHelpers.logger import get_session_id, get_logger

log = get_logger(__name__)

_lock = threading.Lock()
_tracker: dict[str, list[str]] = {}


def record(mem_id: str) -> None:
    if not mem_id:
        return
    session_id = get_session_id()
    if not session_id or session_id == "no-session":
        # Nothing to attribute this to (e.g. called before any session
        # started, such as in a standalone script/test) — skip silently.
        return
    with _lock:
        _tracker.setdefault(session_id, []).append(mem_id)


def get_and_clear(session_id: str) -> list[str]:
    if not session_id:
        return []
    with _lock:
        return _tracker.pop(session_id, [])
