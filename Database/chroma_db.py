import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

_base = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_base)
_cache = os.path.join(_project_root, ".cache", "st")
os.makedirs(_cache, exist_ok=True)
os.environ["SENTENCE_TRANSFORMERS_HOME"] = _cache
os.environ["HF_HOME"] = _cache

from VectorDBClient.ChromaClient import ChromaClient
from dotenv import load_dotenv
import threading

load_dotenv()

semantic_memory_db = None
_chroma_ready = threading.Event()

def _init_chroma():
    global semantic_memory_db
    try:
        semantic_memory_db = ChromaClient(
            collection_name="semantic_memory",
            persist_dir=os.getenv("DEFAULT_PERSIST_DIR"),
            embedding_model=os.getenv("DEFAULT_EMBEDDING_MODEL"),
            distance_fn="cosine",
        )
        # print(f"[ChromaClient] 'semantic_memory' ready — {semantic_memory_db.count()} memories")
    except Exception as e:
        print(f"[chroma_db] FATAL: Could not initialise ChromaDB: {e}")
    finally:
        _chroma_ready.set()  # always signal, even on failure

threading.Thread(target=_init_chroma, daemon=True, name="ChromaInit").start()


def wait_for_chroma(timeout: float = 120.0) -> bool:
    """
    Block until ChromaDB is ready. Call this before the first query,
    not at import time. Returns True if ready, False if timed out.
    """
    return _chroma_ready.wait(timeout=timeout)