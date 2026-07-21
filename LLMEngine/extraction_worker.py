"""
LLMEngine/extraction_worker.py

Owns the background memory-extraction pipeline: a queue that batches
conversation turns and flushes them to semantic-memory extraction on a
cooldown/max-wait schedule, running on its own daemon thread with
contextvars propagated (so session-tagged logging still works from the
background thread — see GlobalHelpers/logger.py's session_id contextvar).

Split out of llm_client.py, where this queue/worker logic was mixed in
with process bootstrap, HTTP request building, and history management.

SHUTDOWN NOTE: because this runs on a daemon thread, anything still
sitting in the batching window (hasn't hit MIN_EXTRACTION_INTERVAL or
MAX_BATCH_WAIT yet) is silently lost if the process exits normally —
the thread just dies with it. flush_and_wait() exists specifically to
be called during an orderly shutdown (see LLMEngine/cli.py), forcing
whatever's pending to extract immediately and blocking until it's
actually done, BEFORE the LLM server subprocess gets stopped.
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

# Set by flush_and_wait() to force an immediate extraction regardless of
# the normal cooldown/max-wait schedule.
_flush_requested = threading.Event()
# Set whenever there is nothing pending and nothing queued — i.e. it is
# safe to assume everything queued so far has actually been extracted.
_drained = threading.Event()
_drained.set()  # nothing pending at import time

MIN_EXTRACTION_INTERVAL = 30.0
MAX_BATCH_WAIT = 90.0  # force-flush safety valve — see review notes on the
                        # original drop-vs-batch extraction bug
_POLL_INTERVAL = 1.0   # how often the worker wakes up to re-check flush/timing


def queue_turn(user_message: str, assistant_activity: str) -> None:
    """Call this once per completed turn. The worker thread handles
    batching/cooldown internally — this just enqueues."""
    _drained.clear()
    _extraction_queue.put((user_message, assistant_activity))


def flush_and_wait(timeout: float = 120.0) -> bool:
    """
    Forces any queued/pending turns to be extracted immediately, ignoring
    the normal cooldown/max-wait batching schedule, and blocks until the
    worker thread has actually finished extracting everything (or until
    `timeout` seconds pass).

    Call this during shutdown — /stop, Ctrl+C, or any other exit path in
    LLMEngine/cli.py — and call it BEFORE stopping the LLM server
    subprocess. extract_and_store_batch() needs the server to still be
    running; flushing after the server's already been stopped just moves
    the same lost-work problem one step later.

    Returns True if fully drained before the timeout, False otherwise
    (logged as a warning either way it's used, not treated as fatal).
    """
    _flush_requested.set()
    finished = _drained.wait(timeout=timeout)
    if not finished:
        log.warning(
            "extraction_worker.flush_and_wait timed out after %.0fs — "
            "some pending turns may not have been extracted before shutdown.",
            timeout,
        )
    _flush_requested.clear()
    return finished


def _extraction_worker() -> None:
    global _last_extraction_time, _first_pending_time
    while True:
        try:
            turn = _extraction_queue.get(timeout=_POLL_INTERVAL)
            got_item = True
        except queue.Empty:
            got_item = False

        if got_item:
            with _pending_lock:
                _pending_batch.append(turn)
                if _first_pending_time is None:
                    _first_pending_time = time.time()

        with _pending_lock:
            has_pending = bool(_pending_batch)

        if not has_pending:
            if got_item:
                _extraction_queue.task_done()
            if _extraction_queue.empty():
                _drained.set()
            continue

        now = time.time()
        cooldown_elapsed = now - _last_extraction_time >= MIN_EXTRACTION_INTERVAL
        waited_too_long = (
            _first_pending_time is not None
            and now - _first_pending_time >= MAX_BATCH_WAIT
        )
        force_flush = _flush_requested.is_set()

        if not (cooldown_elapsed or waited_too_long or force_flush):
            if got_item:
                _extraction_queue.task_done()
            continue

        if force_flush:
            # Drain anything else already sitting in the queue too, so a
            # shutdown flush combines the whole tail of the session into
            # ONE extraction call instead of fragmenting it into several
            # single-turn calls (each item would otherwise trigger its own
            # immediate force-extract as soon as it's dequeued).
            while True:
                try:
                    extra_turn = _extraction_queue.get_nowait()
                except queue.Empty:
                    break
                with _pending_lock:
                    _pending_batch.append(extra_turn)
                _extraction_queue.task_done()

        with _pending_lock:
            batch, _pending_batch[:] = list(_pending_batch), []
            _first_pending_time = None

        extract_and_store_batch(batch)
        _last_extraction_time = time.time()
        if got_item:
            _extraction_queue.task_done()

        with _pending_lock:
            still_pending = bool(_pending_batch)
        if not still_pending and _extraction_queue.empty():
            _drained.set()


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
