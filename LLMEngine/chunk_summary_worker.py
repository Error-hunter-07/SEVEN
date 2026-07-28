"""
LLMEngine/chunk_summary_worker.py

Background worker for rolling episodic chunk summarization. Every 5
turns (LLMEngine/llm_client.py checks the turn count after each
heartbeat), the last 5 turns get queued here and summarized on this
dedicated daemon thread — not on the main thread, and not synchronously
inside ask_llm() — so the user's next message is never blocked waiting
for a summarization call to finish.

Mirrors LLMEngine/extraction_worker.py's shape: a queue + one daemon
thread + contextvars propagated at start() so session-tagged logging
still works correctly from this background thread (see
GlobalHelpers/logger.py's session_id contextvar).

Unlike extraction_worker.py, this has NO cooldown/batching logic — the
"every 5 turns" gate already happens at the call site (llm_client.py),
so by the time something reaches this queue it's meant to be processed
as its own distinct chunk, not coalesced with whatever comes next.
Coalescing chunk summaries together would lose the "in order" narrative
structure summarize_session() depends on when it stitches them back
together at session end.

Every call into the local LLM server (via summarizer.summarize_chunk())
goes through LLMEngine.llm_request_lock, so this worker never races the
main chat turn or the semantic-memory extraction worker for the local
server's single processing slot (--parallel 1).
"""

import queue
import threading
import contextvars

import MemoryManagement.episodic_memory.summarizer as summarizer
import Database.active_sessions_db_client as active_sessions_db_client
from GlobalHelpers.logger import get_logger

log = get_logger(__name__)

_chunk_queue: "queue.Queue[tuple[str, list[tuple[str, str]]]]" = queue.Queue()
_worker_lock = threading.Lock()
_worker_started = False


def queue_chunk(session_id: str, turns: list[tuple[str, str]]) -> None:
    """Call this once every 5 turns (see llm_client.py's heartbeat
    check). `turns` is a list of (user_message, assistant_reply) pairs
    covering just this chunk — the worker summarizes them in order and
    appends the result to active_sessions.chunk_summaries for
    `session_id`. Non-blocking — just enqueues."""
    if not session_id or not turns:
        return
    _chunk_queue.put((session_id, turns))


def _chunk_worker() -> None:
    while True:
        session_id, turns = _chunk_queue.get()
        try:
            summary = summarizer.summarize_chunk(turns)
            if summary:
                active_sessions_db_client.append_chunk_summary(session_id, summary)
                log.debug("Chunk summary appended for session %s.", session_id)
            else:
                log.warning(
                    "Chunk summarization returned nothing for session %s — "
                    "full_conversation backup still covers this slice.",
                    session_id,
                )
        except Exception:
            log.exception(
                "Chunk summarization failed (non-fatal — full_conversation backup still covers this slice)."
            )
        finally:
            _chunk_queue.task_done()


def start() -> None:
    """Starts the background worker thread exactly once, propagating the
    current contextvars so log lines from this thread stay tagged with
    the right session_id instead of falling back to 'no-session'."""
    global _worker_started
    with _worker_lock:
        if _worker_started:
            return
        ctx = contextvars.copy_context()
        threading.Thread(
            target=lambda: ctx.run(_chunk_worker),
            daemon=True,
            name="ChunkSummary",
        ).start()
        _worker_started = True