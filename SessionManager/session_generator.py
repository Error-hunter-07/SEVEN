from __future__ import annotations
import uuid
import threading
from typing import Set

# /c:/Users/singh/OneDrive/Desktop/Seven/SessionGenerator/session_generator.py
"""
Generate universally-unique session IDs (non-repeating within and across runs
with extremely high probability). Thread-safe.
"""


_lock = threading.Lock()
_generated_ids: Set[str] = set()


def generate_universal_session_id() -> str:
    """
    Generate a universally-unique session id as a 32-character hex string.
    Keeps an in-memory registry to avoid repeats within this process and
    retries on collision. Thread-safe.
    """
    with _lock:
        # primary: random UUID4 (extremely low collision probability)
        for _ in range(8):
            uid = uuid.uuid4().hex
            if uid not in _generated_ids:
                _generated_ids.add(uid)
                return uid
        # fallback: use UUID1 (time+node) until unique
        while True:
            uid = uuid.uuid1().hex
            if uid not in _generated_ids:
                _generated_ids.add(uid)
                return uid