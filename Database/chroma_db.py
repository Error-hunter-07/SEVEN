import os
import logging

os.environ["CUDA_VISIBLE_DEVICES"] = ""

_base = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_base)
_cache = os.path.join(_project_root, ".cache", "st")
os.makedirs(_cache, exist_ok=True)
os.environ["SENTENCE_TRANSFORMERS_HOME"] = _cache
os.environ["HF_HOME"] = _cache

# Suppress progress bars from sentence-transformers/transformers
# This must be set globally at module level, before ChromaClient init
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

from VectorDBClient.ChromaClient import ChromaClient
import threading
from GlobalHelpers.logger import get_logger
from GlobalHelpers.config import settings

log = get_logger(__name__)

semantic_memory_db = None
episodic_memory_db = None
_chroma_ready = threading.Event()


def _init_chroma():
    """
    Initializes BOTH collections in one thread, gated by one readiness
    event. They share init because the app isn't meaningfully usable
    until both are up anyway (episodic memory's passive 2-session seed
    runs at every session start, right alongside semantic memory's
    goals/experience seed) — no benefit to tracking their readiness
    separately, and one shared event keeps callers simple (one
    wait_for_chroma() instead of two near-identical waits).
    """
    global semantic_memory_db, episodic_memory_db
    try:
        semantic_memory_db = ChromaClient(
            collection_name="semantic_memory",
            persist_dir=settings.default_persist_dir,
            embedding_model=settings.default_embedding_model,
            distance_fn="cosine",
        )
        episodic_memory_db = ChromaClient(
            collection_name="episodic_memory",
            persist_dir=settings.default_persist_dir,
            embedding_model=settings.default_embedding_model,
            distance_fn="cosine",
        )
    except Exception as e:
        log.critical("FATAL: Could not initialise ChromaDB: %s", e, exc_info=True)
    finally:
        _chroma_ready.set()  # always signal, even on failure

threading.Thread(target=_init_chroma, daemon=True, name="ChromaInit").start()


def wait_for_chroma(timeout: float = 120.0) -> bool:
    """
    Block until both ChromaDB collections are ready. Call this before
    the first query, not at import time. Returns True if ready, False
    if timed out.
    """
    return _chroma_ready.wait(timeout=timeout)