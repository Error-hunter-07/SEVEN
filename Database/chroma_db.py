import os
import logging

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"
os.environ["CHROMA_ANONYMIZED_TELEMETRY"] = "FALSE"

_base = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_base)
_cache = os.path.join(_project_root, ".cache", "st")
os.makedirs(_cache, exist_ok=True)
os.environ["SENTENCE_TRANSFORMERS_HOME"] = _cache
os.environ["HF_HOME"] = _cache

# CHANGED: Stop sentence-transformers from pinging HuggingFace Hub on
# every startup to check whether cached model files are current.
#
# Root cause: huggingface_hub fires HEAD requests for every model file
# on every load to validate the local cache against the server — 20+
# network round trips even when nothing has changed. This adds ~10s to
# startup and breaks entirely when offline.
#
# Fix: set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 unconditionally
# so the HF library never touches the network. Both flags must be set
# here, at module level, BEFORE any HF/sentence-transformers import
# happens — setting them after the library is imported has no effect
# because huggingface_hub reads these flags once at import time.
#
# First-run handling: if the model is not yet cached, the offline load
# will raise huggingface_hub.errors.OfflineModeIsEnabled (or OSError on
# older versions). _init_chroma() catches this, temporarily lifts the
# flag, downloads the model, then re-enables offline mode so the NEXT
# startup (and all future ones) are instant.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Suppress progress bars and noisy logs from sentence-transformers /
# transformers. Set before any import of those libraries.
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


def _make_chroma_client(collection_name: str) -> ChromaClient:
    """
    Creates one ChromaClient, with an automatic online-fallback on first
    run. Two attempts:

      1. Offline (HF_HUB_OFFLINE=1 already set): fast, no network.
         Succeeds on every run after the first download.

      2. Online fallback: only reached if attempt 1 raises an offline
         or missing-model error, which only happens when the model has
         never been downloaded to this machine. We temporarily clear the
         offline flags, let the download complete, then re-set them so
         the next startup is offline again.

    Keeping the two-attempt logic here (rather than at module level)
    means each collection independently handles a partial cache — if
    somehow only one collection's embedding model is cached, the other
    can still download without affecting the first.
    """
    try:
        return ChromaClient(
            collection_name=collection_name,
            persist_dir=settings.default_persist_dir,
            embedding_model=settings.default_embedding_model,
            distance_fn="cosine",
        )
    except Exception as first_err:
        # Distinguish a genuine offline-model-missing error from any
        # other init failure (e.g. disk full, bad collection name).
        # huggingface_hub raises OfflineModeIsEnabled; older sentence-
        # transformers surfaces this as an OSError with "offline" or
        # "No such file" in the message. We match broadly rather than
        # importing the specific exception class so this works across
        # library versions.
        err_str = str(first_err).lower()
        is_offline_miss = (
            "offline" in err_str
            or "no such file" in err_str
            or "not found in cache" in err_str
            or "localentrynotfounderror" in err_str
            or "offlinemodeenabled" in type(first_err).__name__.lower()
        )
        if not is_offline_miss:
            raise  # not a cache-miss — re-raise immediately

        log.warning(
            "_make_chroma_client: model not in local cache (%s). "
            "Falling back to online download — this only happens once.",
            first_err,
        )
        # Temporarily lift offline restriction for the download
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        try:
            client = ChromaClient(
                collection_name=collection_name,
                persist_dir=settings.default_persist_dir,
                embedding_model=settings.default_embedding_model,
                distance_fn="cosine",
            )
            log.info(
                "_make_chroma_client: model downloaded successfully. "
                "All future startups will use the local cache (offline mode)."
            )
            return client
        finally:
            # Always re-enable offline mode, whether the download
            # succeeded or failed — don't leave the process in an
            # unintended online state.
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _init_chroma():
    """
    Initializes BOTH collections in one thread, gated by one readiness
    event. They share init because the app isn't meaningfully usable
    until both are up anyway — no benefit to tracking their readiness
    separately, and one shared event keeps callers simple.
    """
    global semantic_memory_db, episodic_memory_db
    try:
        semantic_memory_db = _make_chroma_client("semantic_memory")
        episodic_memory_db = _make_chroma_client("episodic_memory")
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