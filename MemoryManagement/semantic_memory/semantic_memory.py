"""
High-level semantic memory API.
Everything above this layer (tools, retriever, extractor) uses this class.
Everything below this layer (ChromaDB) is hidden inside Database/chroma_db.py.

This class never imports chromadb directly — only VectorDBClient.
"""

from __future__ import annotations
import threading

from Database.chroma_db import semantic_memory_db
from VectorDBClient.VectorClient import VectorDBClient

from MemoryManagement.semantic_memory import memory_lifecycle
from GlobalHelpers.logger import get_logger

log = get_logger(__name__)

class SemanticMemory:

    def __init__(self, db: VectorDBClient | None = None):
        # Allow injecting a different backend (useful for tests)
        self._db = db or semantic_memory_db
        # Don't start lifecycle immediately — ChromaDB may still be loading
        threading.Thread(target=self._start_lifecycle_when_ready, daemon=True).start()

    def _start_lifecycle_when_ready(self):
        import Database.chroma_db as chroma_module
        ready = chroma_module.wait_for_chroma(timeout=120)
        if not ready or chroma_module.semantic_memory_db is None:
            log.warning("ChromaDB never became ready — lifecycle skipped.")
            return
        memory_lifecycle.start(chroma_module.semantic_memory_db)
    # ---------------------------------------------------------------- write

    def store(
        self,
        text: str,
        importance: float = 0.5,
        category: str     = "other",
        source: str       = "conversation",
    ) -> str | None:
        """
        Distil and store a single memory fact.

        Before inserting, checks for near-duplicates (cosine dist <= 0.08).
        If one exists, bumps its access_count instead of creating a new entry.

        Returns the memory id on success, None on failure.
        """
        if self._db is None:
            log.warning("store: DB not available.")
            return None

        text = text.strip()
        if not text:
            return None

        # overflow guard on every store

        if self._db.count() > memory_lifecycle.MAX_MEMORIES:
            memory_lifecycle._prune(self._db)  

            
        # Deduplication check
        # Stage 1: near-identical (current threshold) → update access count only
        near_duplicate = self._db.find_duplicate(text, threshold=0.08)
        if near_duplicate:
            mem_id   = near_duplicate["id"]
            old_meta = near_duplicate["metadata"]
            self._db.update(
                id=mem_id,
                metadata={
                    **old_meta,
                    "access_count":  int(old_meta.get("access_count", 0)) + 1,
                    "last_accessed": VectorDBClient.now_iso(),
                    # Update importance if the new value is higher
                    "importance": max(float(old_meta.get("importance", 0.5)), importance),
                },
            )
            log.debug("Dedup hit — updated existing memory: %s", mem_id)
            return mem_id
        
        
        # Stage 2: paraphrase range (0.08–0.35) → only merge if SAME category
        paraphrase_dup = self._db.find_duplicate(text, threshold=0.35)
        if paraphrase_dup:
            same_category = paraphrase_dup["metadata"].get("category") == category
            if same_category:
                # Keep the higher-importance version's text, merge metadata
                old_importance = float(paraphrase_dup["metadata"].get("importance", 0.5))
                if importance > old_importance:
                    # New version is more important — replace text, keep id
                    self._db.update(
                        id=paraphrase_dup["id"],
                        text=text,  # upgrade to better phrasing
                        metadata={**paraphrase_dup["metadata"], "importance": importance}
                    )
                else:
                    # Old version stays, just bump count
                    self._db.update(id=paraphrase_dup["id"], metadata={
                        **paraphrase_dup["metadata"],
                        "access_count": int(paraphrase_dup["metadata"].get("access_count", 0)) + 1
                    })
                return paraphrase_dup["id"]

        # New memory
        mem_id = VectorDBClient.new_id()
        now    = VectorDBClient.now_iso()
        success = self._db.add(
            id=mem_id,
            text=text,
            metadata={
                "importance":    importance,
                "category":      category,
                "source":        source,
                "created_at":    now,
                "last_accessed": now,
                "access_count":  0,
            },
        )
        if success:
            log.debug("Stored memory [%s] %s", category, text[:80])
            return mem_id
        return None

    def delete(self, memory_id: str) -> bool:
        if self._db is None:
            return False
        return self._db.delete(memory_id)

    # ---------------------------------------------------------------- read

    def retrieve(
        self,
        query: str,
        k: int             = 5,
        min_importance: float | None = None,
        category: str | None         = None,
        update_access: bool = False,   # FIX 1: default False — don't write on reads
    ) -> list[dict]:
        """
        Return the k most relevant memories for `query`.

        Args:
            query:          The current user message or topic to search against.
            k:              Max number of memories to return.
            min_importance: Optional floor — only return memories with
                            importance >= this value.
            category:       Optional filter — only return memories in this category.

        Returns:
            List of dicts: [{id, text, metadata, score}, ...]
            Sorted by relevance (lowest cosine distance first).
        """
        if self._db is None:
            return []

        where = self._build_where(min_importance, category)
        results = self._db.search(query=query, k=k, where=where)

        # FIX 1: Only update access metadata when explicitly requested.
        # Previously this fired on EVERY turn (5 writes per prompt build)
        # which was blocking the main thread and causing empty responses.
        # Now only called when the LLM explicitly searches via tool call.
        if update_access:
            for r in results:
                meta = r["metadata"]
                self._db.update(
                    id=r["id"],
                    metadata={
                        **meta,
                        "last_accessed": VectorDBClient.now_iso(),
                        "access_count": int(meta.get("access_count", 0)) + 1,
                    },
                )
 
        return results

    def retrieve_as_text(
        self,
        query: str,
        k: int                       = 5,
        min_importance: float | None = None,
        category: str | None         = None,
    ) -> str:
        """
        Same as retrieve() but returns a formatted string ready to inject
        into the system prompt (used by memory_retriever.py).
        """
        memories = self.retrieve(query=query, k=k, min_importance=min_importance, category=category, update_access=False)   # never write during prompt building

        RELEVANCE_THRESHOLD = 0.5
        
        memories = [m for m in memories if m["score"] <= RELEVANCE_THRESHOLD]
        if not memories:
            return ""

        lines = ["LONG-TERM MEMORY (semantic):"]
        for i, m in enumerate(memories, 1):
            meta     = m["metadata"]
            mem_category = meta.get("category", "")      # FIX 3: renamed to avoid shadowing parameter
            imp      = meta.get("importance", "")
            lines.append(f"  {i}. [{mem_category}] {m['text']}  (importance: {imp})")
        return "\n".join(lines)

    def count(self) -> int:
        if self._db is None:
            return 0
        return self._db.count()

    # ---------------------------------------------------------------- internal

    @staticmethod
    def _build_where(
        min_importance: float | None,
        category: str | None,
    ) -> dict | None:
        """Build a ChromaDB `where` filter from optional constraints."""
        filters = []
        if min_importance is not None:
            filters.append({"importance": {"$gte": min_importance}})
        if category:
            filters.append({"category": {"$eq": category}})

        if len(filters) == 0:
            return None
        if len(filters) == 1:
            return filters[0]
        return {"$and": filters}


# FIX 2: Module-level singleton — all three files (memory_retriever,
# semantic_memory_tool, memory_extractor) import THIS instead of calling
# SemanticMemory() themselves. One instance, one ChromaDB connection.
semantic_memory = SemanticMemory()