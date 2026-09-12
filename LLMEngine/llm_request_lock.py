"""
LLMEngine/llm_request_lock.py

The local llama-server runs --parallel 1: it can only physically process
one request at a time, no matter how many Python threads try to call it
concurrently. Without a shared lock, two callers hitting the server at
close to the same moment (e.g. the main chat turn and a background
chunk-summary job) queue at the server unpredictably, and worse, can
thrash llama.cpp's per-slot prompt cache (evicting a checkpoint one
caller was relying on before it gets used again — see server log
evidence of "restored context checkpoint" vs "forcing full prompt
re-processing").

This lock makes that contention explicit and bounded instead of an
implicit race: every LLM HTTP call in the app — the main chat
completion, the semantic-memory extractor, the episodic summarizer/
chunk-summarizer — acquires a lock before calling requests.post, so at
most one request per role is ever in flight.

CHANGED (background mini-LLM): role-aware instead of one single global
endpoint/lock. The main chat model (role="main", the default — existing
callers don't need to change anything) and the background mini-LLM
(role="background", see Runtime/background_process.py) now run as two
separate llama-server processes on two separate ports, each with its OWN
lock. That's what actually removes the contention rather than just
making it safe: the main chat turn and a background extraction/
summarization call can now genuinely run at the same time, since they're
not sharing a lock *or* a GPU slot anymore. Within a single role, the
lock still applies exactly as before — e.g. two background calls
(extraction + chunk-summary) racing each other still queue safely behind
role="background"'s own lock.

If this project ever moves to --parallel 2+ with slot pinning
(id_slot), each role's lock can be relaxed to one-lock-per-slot instead
of one-lock-per-role — not needed until then.
"""

import threading
import requests
from GlobalHelpers.logger import get_logger

log = get_logger(__name__)

# (endpoint, lock) per role. Ports must match Runtime/main_process.py's
# PORT and Runtime/background_process.py's PORT respectively — kept as
# plain constants here rather than importing those modules to avoid
# pulling in Runtime's config/settings dependency from this low-level
# HTTP module; if either port ever changes, update both places.
_ENDPOINTS = {
    "main":       ("http://127.0.0.1:8081/v1/chat/completions", threading.Lock()),
    "background": ("http://127.0.0.1:8082/v1/chat/completions", threading.Lock()),
}


def post_completion(payload: dict, role: str = "main", timeout: float = 120.0) -> requests.Response:
    """
    Every LLM call in the app should go through this function instead of
    calling requests.post directly, so the shared lock is never
    accidentally bypassed by a new call site added later.

    role: "main" (default — main chat completion, LLMEngine/llm_client.py)
          or "background" (semantic-memory extraction and episodic/chunk
          summarization — see MemoryManagement/semantic_memory/
          memory_extractor.py and MemoryManagement/episodic_memory/
          summarizer.py). Existing callers that don't pass role at all
          keep hitting role="main" exactly as before this change.
    """
    if role not in _ENDPOINTS:
        raise ValueError(
            f"Unknown LLM role {role!r} — expected one of {sorted(_ENDPOINTS)}."
        )
    url, lock = _ENDPOINTS[role]
    with lock:
        return requests.post(url, json=payload, timeout=timeout)