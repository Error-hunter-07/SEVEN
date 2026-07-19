"""
LLMEngine/extraction_worker.py

Owns the background memory-extraction pipeline: a queue that batches
conversation turns and flushes them to semantic-memory extraction on a
cooldown/max-wait schedule, running on its own daemon thread with
contextvars propagated (so session-tagged logging still works from the
background thread — see GlobalHelpers/logger.py's session_id contextvar).

Split out of llm_client.py, where this queue/worker logic was mixed in
with process bootstrap, HTTP request building, and history management.
"""

import queue
import threading
import time
import contextvars

from MemoryManagement.semantic_memory.memory_extractor import extract_and_store_batch
from GlobalHelpers.logger import get_logger

log = get_logger(__name__)

_extraction_queue: "queue.Queue[tuple[str, str]]" = queue.Queue()
_pending_batch: list[tuple[str, str]] = []
_pending_lock = threading.Lock()
_worker_lock = threading.Lock()
_worker_started = False
_last_extraction_time = 0.0
_first_pending_time = None

MIN_EXTRACTION_INTERVAL = 30.0
MAX_BATCH_WAIT = 90.0  # force-flush safety valve — see review notes on the
                        # original drop-vs-batch extraction bug


def queue_turn(user_message: str, assistant_activity: str) -> None:
    """Call this once per completed turn. The worker thread handles
    batching/cooldown internally — this just enqueues."""
    _extraction_queue.put((user_message, assistant_activity))


def _extraction_worker() -> None:
    global _last_extraction_time, _first_pending_time
    while True:
        turn = _extraction_queue.get()
        with _pending_lock:
            _pending_batch.append(turn)
            if _first_pending_time is None:
                _first_pending_time = time.time()

        now = time.time()
        cooldown_elapsed = now - _last_extraction_time >= MIN_EXTRACTION_INTERVAL
        waited_too_long = (
            _first_pending_time is not None
            and now - _first_pending_time >= MAX_BATCH_WAIT
        )

        if not (cooldown_elapsed or waited_too_long):
            _extraction_queue.task_done()
            continue

        with _pending_lock:
            batch, _pending_batch[:] = list(_pending_batch), []
            _first_pending_time = None

        extract_and_store_batch(batch)
        _last_extraction_time = time.time()
        _extraction_queue.task_done()


def start() -> None:
    """Starts the background worker thread exactly once, propagating the
    current contextvars so log lines from the worker thread stay tagged
    with the right session_id instead of falling back to 'no-session'."""
    global _worker_started
    with _worker_lock:
        if _worker_started:
            return
        ctx = contextvars.copy_context()
        threading.Thread(
            target=lambda: ctx.run(_extraction_worker),
            daemon=True,
            name="MemoryExtraction",
        ).start()
        _worker_started = True
