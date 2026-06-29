import os
# FIX 1: Force SentenceTransformer to CPU only.
# Without this, PyTorch grabs CUDA context before llama-server can,
# causing VRAM contention and making llama-server fail silently.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
 
# FIX 2: Use an ABSOLUTE cache path so the model is never re-downloaded.
# The relative path '.cache/st' was resolving differently each run
# depending on working directory, causing a fresh 90MB download every startup.
_base = os.path.dirname(os.path.abspath(__file__))          # .../Seven/Database/
_project_root = os.path.dirname(_base)                       # .../Seven/
_cache = os.path.join(_project_root, ".cache", "st")
os.makedirs(_cache, exist_ok=True)
os.environ["SENTENCE_TRANSFORMERS_HOME"] = _cache
os.environ["HF_HOME"] = _cache                               # some versions use this
 
from VectorDBClient.ChromaClient import ChromaClient
from dotenv import load_dotenv
load_dotenv()
 
try:
    semantic_memory_db = ChromaClient(
        collection_name="semantic_memory",
        persist_dir=os.getenv("DEFAULT_PERSIST_DIR"),
        embedding_model=os.getenv("DEFAULT_EMBEDDING_MODEL"),
        distance_fn="cosine",
    )
except Exception as e:
    print(f"[chroma_db] FATAL: Could not initialise ChromaDB: {e}")
    semantic_memory_db = None

