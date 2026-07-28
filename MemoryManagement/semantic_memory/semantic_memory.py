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
import SessionManager.session_memory_tracker as session_memory_tracker
from GlobalHelpers.logger import get_logger

log = get_logger(__name__)

# ── Negation markers for quick polarity mismatch detection ────────────────────
_NEGATION_MARKERS = (
    "not", "n't", "never", "no longer", "stopped", "quit",
    "dislikes", "dislike", "hates", "hate", "doesn't", "don't", "without"
)


def _has_negation_mismatch(text_a: str, text_b: str) -> bool:
    """
    Cheap heuristic guard: if exactly one of the two texts contains a negation
    marker, treat them as opposites, not paraphrases — never merge.
    
    This catches common cases like "likes X" vs "dislikes X" for near-zero cost.
    Not linguistically bulletproof, but prevents the worst failure mode.
    """
    a_neg = any(marker in text_a.lower() for marker in _NEGATION_MARKERS)
    b_neg = any(marker in text_b.lower() for marker in _NEGATION_MARKERS)
    return a_neg != b_neg  # True only when they disagree on negation


class SemanticMemory:

    def __init__(self, db: VectorDBClient | None = None):
        # Allow injecting a different backend (useful for tests)
        self._db = db or semantic_memory_db
        self._injected = db is not None
        # Don't start lifecycle immediately — ChromaDB may still be loading
        threading.Thread(target=self._start_lifecycle_when_ready, daemon=True).start()

    def _start_lifecycle_when_ready(self):
        import Database.chroma_db as chroma_module
        ready = chroma_module.wait_for_chroma(timeout=120)
        if not ready or chroma_module.semantic_memory_db is None:
            log.warning("ChromaDB never became ready — lifecycle skipped.")
            return
        if not self._injected:
            self._db = chroma_module.semantic_memory_db
        memory_lifecycle.start(chroma_module.semantic_memory_db)
    # ---------------------------------------------------------------- write

    def store(
        self,
        text: str,
        importance: float = 0.5,
        category: str     = "other",
        polarity: str     = "neutral",
        source: str       = "conversation",
    ) -> str | None:
        """
        Distil and store a single memory fact.

        Before inserting, checks for near-duplicates (cosine dist <= 0.08).
        If one exists, bumps its access_count instead of creating a new entry.
        
        Checks for paraphrase-range duplicates (0.08–0.35) and merges only if
        they have the same category AND polarity AND no negation mismatch.

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
        
        
        # Stage 2: paraphrase range (0.08–0.35) → only merge if SAME category, polarity, and no negation mismatch
        paraphrase_dup = self._db.find_duplicate(text, threshold=0.35)
        if paraphrase_dup:
            same_category = paraphrase_dup["metadata"].get("category") == category
            same_polarity = paraphrase_dup["metadata"].get("polarity", "neutral") == polarity
            contradicts = _has_negation_mismatch(text, paraphrase_dup["text"])
            
            if same_category and same_polarity and not contradicts:
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
            # else: contradicts or mismatch — fall through to "New memory" below

        # New memory
        mem_id = VectorDBClient.new_id()
        now    = VectorDBClient.now_iso()
        success = self._db.add(
            id=mem_id,
            text=text,
            metadata={
                "importance":    importance,
                "category":      category,
                "polarity":      polarity,
                "source":        source,
                "created_at":    now,
                "last_accessed": now,
                "access_count":  0,
            },
        )
        if success:
            log.debug("Stored memory [%s] %s", category, text[:80])
            # Only genuinely NEW memories get recorded here — the two
            # dedup paths above return early with an EXISTING id, which
            # likely already belongs to some earlier episode's
            # related_semantic_memory_ids, so re-attributing it to
            # whichever session happened to touch it again would make
            # that link noisy rather than useful.
            session_memory_tracker.record(mem_id)
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

    def get_by_ids(self, ids: list[str]) -> list[dict]:
        """
        Fetch specific memories by id, no search involved. Used to
        resolve an episode's related_semantic_memory_ids into actual
        fact text/metadata — e.g. the episodic search tool compiling
        "here's the episode, and here are the facts tied to it" in one
        response.

        Silently drops any id that no longer exists (a linked semantic
        memory can outlive its own dedup/prune cycle independent of the
        episode that referenced it) rather than erroring on a partial
        miss.
        """
        if self._db is None or not ids:
            return []
        if not hasattr(self._db, "get_many"):
            # Backend doesn't support batch fetch (e.g. a test double) —
            # fall back to one get() per id.
            results = []
            for mem_id in ids:
                r = self._db.get(mem_id)
                if r is not None:
                    results.append(r)
            return results
        return self._db.get_many(ids)

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