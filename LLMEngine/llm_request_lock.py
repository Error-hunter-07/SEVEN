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
chunk-summarizer — acquires this same lock before calling requests.post,
so at most one request is ever in flight. Combined with keeping
background calls short (low max_tokens), the worst-case delay this adds
to the user's next message is small and predictable rather than
unbounded.

If this project ever moves to --parallel 2+ with slot pinning
(id_slot), this lock can be relaxed to one-lock-per-slot instead of a
single global lock — not needed until then.
"""

import threading
import requests
from GlobalHelpers.logger import get_logger

log = get_logger(__name__)

_llm_lock = threading.Lock()

LLM_ENDPOINT = "http://127.0.0.1:8081/v1/chat/completions"


def post_completion(payload: dict, timeout: float = 120.0) -> requests.Response:
    """
    Every LLM call in the app should go through this function instead of
    calling requests.post directly, so the shared lock is never
    accidentally bypassed by a new call site added later.
    """
    with _llm_lock:
        return requests.post(LLM_ENDPOINT, json=payload, timeout=timeout)