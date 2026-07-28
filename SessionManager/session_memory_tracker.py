"""
SessionManager/session_memory_tracker.py

CHANGED: previously a plain in-memory {session_id: [mem_ids]} dict, with
a documented "acceptable gap" — if the process crashed before
on_session_end ran, this tracker's contents were simply lost, and a
crash-recovered episode's related_semantic_memory_ids was left empty.

Now delegates to Database/active_sessions_db_client.py, which persists
this list into the active_sessions row for the current session on every
write. This closes that gap: a crash-recovered episode can now include
real related_semantic_memory_ids instead of none, because the list
survives the crash the same way chunk_summaries and full_conversation
already do.

record()/get_and_clear() keep their exact original signatures and call
sites (semantic_memory.py's store(), session_lifecycle.py's
on_session_end) — only the storage underneath moved from an in-memory
dict to the durable active_sessions table. This module still exists as
a thin wrapper (rather than every caller reaching into
active_sessions_db_client directly) so callers keep resolving "which
session is this" via get_session_id() exactly as before.

Keyed off GlobalHelpers.logger.get_session_id() rather than a value
passed around explicitly, because that contextvar already correctly
propagates into background threads (the batch-extraction worker, and
now the chunk-summary worker too) via contextvars.copy_context() — see
LLMEngine/extraction_worker.py. That means this tracker works correctly
for writes from any of the call sites that create semantic memory
(background extraction, the LLM's direct tool call, and
session_lifecycle's own promotion writes), without threading
id-collection through each of them individually.
"""

from GlobalHelpers.logger import get_session_id, get_logger
import Database.active_sessions_db_client as active_sessions_db_client

log = get_logger(__name__)


def record(mem_id: str) -> None:
    if not mem_id:
        return
    session_id = get_session_id()
    if not session_id or session_id == "no-session":
        # Nothing to attribute this to (e.g. called before any session
        # started, such as in a standalone script/test) — skip silently.
        return
    active_sessions_db_client.append_semantic_memory_id(session_id, mem_id)


def get_and_clear(session_id: str) -> list[str]:
    """
    NOTE: despite the name, this no longer clears anything itself.
    active_sessions rows are deleted wholesale by close_session() right
    after on_session_end finishes using this list — there's nothing left
    to separately clear here. Keeping the original name for call-site
    compatibility (session_lifecycle.py calls this unchanged) rather
    than renaming for a purely cosmetic reason.
    """
    if not session_id:
        return []
    return active_sessions_db_client.get_related_semantic_memory_ids(session_id)