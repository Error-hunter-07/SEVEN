from VectorDBClient.ChromaClient import ChromaClient
import os
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


