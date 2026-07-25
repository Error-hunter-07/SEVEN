from __future__ import annotations
import os
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from VectorDBClient.VectorClient import VectorDBClient
from GlobalHelpers.logger import get_logger

log = get_logger(__name__)


class ChromaClient(VectorDBClient):
    """
        pip install chromadb sentence-transformers
    """

    def __init__(
        self,
        collection_name: str = "semantic_memory",
        persist_dir: str     = None,
        embedding_model: str = None,
        distance_fn: str     = "cosine",
    ):
        try:
            import chromadb
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        except ImportError as e:
            raise ImportError(
                "Run:  pip install chromadb sentence-transformers"
            ) from e

        self._collection_name = collection_name
        self._distance_fn     = distance_fn

        self._embedding_fn = SentenceTransformerEmbeddingFunction(model_name=embedding_model)
        self._client       = chromadb.PersistentClient(path=persist_dir)
        self._collection   = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": distance_fn},
        )

        log.info("'%s' ready — %d memories", collection_name, self._collection.count())

    # --------------- CRUD Operations ----------------------------------#

    def add(self, id: str, text: str, metadata: dict[str, Any]) -> bool:
        if not text or not text.strip():
            log.warning("add: empty text — skipped.")
            return False
        try:
            self._collection.add(
                ids=[id],
                documents=[text],
                metadatas=[self._sanitise(metadata)],
            )
            return True
        except Exception as e:
            log.error("add error: %s", e, exc_info=True)
            return False

    def search(self, query: str, k: int = 5, where: dict | None = None) -> list[dict]:
        if not query or not query.strip():
            return []
        total = self._collection.count()
        if total == 0:
            return []
        safe_k = min(k, total)
        try:
            kwargs: dict[str, Any] = {
                "query_texts": [query],
                "n_results":   safe_k,
                "include":     ["documents", "metadatas", "distances"],
            }
            if where:
                kwargs["where"] = where
            raw = self._collection.query(**kwargs)
            return [
                {"id": i, "text": d, "metadata": m, "score": round(s, 6)}
                for i, d, m, s in zip(
                    raw["ids"][0],
                    raw["documents"][0],
                    raw["metadatas"][0],
                    raw["distances"][0],
                )
            ]
        except Exception as e:
            log.error("search error: %s", e, exc_info=True)
            return []

    def get(self, id: str) -> dict | None:
        try:
            raw = self._collection.get(ids=[id], include=["documents", "metadatas"])
            if not raw["ids"]:
                return None
            return {
                "id":       raw["ids"][0],
                "text":     raw["documents"][0],
                "metadata": raw["metadatas"][0],
            }
        except Exception as e:
            log.error("get error: %s", e, exc_info=True)
            return None

    def get_many(self, ids: list[str]) -> list[dict]:
        """Fetches multiple specific entries by id in one call. Used by
        SemanticMemory.get_by_ids() to resolve an episode's
        related_semantic_memory_ids into actual fact text."""
        if not ids:
            return []
        try:
            raw = self._collection.get(ids=ids, include=["documents", "metadatas"])
            return [
                {"id": i, "text": d, "metadata": m}
                for i, d, m in zip(raw["ids"], raw["documents"], raw["metadatas"])
            ]
        except Exception as e:
            log.error("get_many error: %s", e, exc_info=True)
            return []

    def get_by_metadata(self, where: dict, limit: int | None = None) -> list[dict]:
        """
        Fetches entries matching a metadata filter with NO query text/vector
        involved — a plain relational-style lookup, not a similarity
        search. Used for things like "all episodes at decay_count=0
        older than X" where there's no meaningful query to embed, only a
        filter condition.

        Chroma's .get() (unlike .query()) doesn't rank by similarity or
        take a query, so this returns matches in whatever order the
        collection yields them — callers that need a specific order
        (e.g. oldest-first for decay batching) must sort the result
        themselves.
        """
        try:
            kwargs: dict[str, Any] = {"where": where, "include": ["documents", "metadatas"]}
            if limit is not None:
                kwargs["limit"] = limit
            raw = self._collection.get(**kwargs)
            return [
                {"id": i, "text": d, "metadata": m}
                for i, d, m in zip(raw["ids"], raw["documents"], raw["metadatas"])
            ]
        except Exception as e:
            log.error("get_by_metadata error: %s", e, exc_info=True)
            return []

    def update(self, id: str, text: str | None = None, metadata: dict | None = None) -> bool:
        existing = self.get(id)
        if existing is None:
            log.warning("update: id '%s' not found.", id)
            return False
        new_text     = (text or existing["text"]).strip()
        merged_meta  = {**existing["metadata"], **(metadata or {})}
        try:
            self._collection.update(
                ids=[id],
                documents=[new_text],
                metadatas=[self._sanitise(merged_meta)],
            )
            return True
        except Exception as e:
            log.error("update error: %s", e, exc_info=True)
            return False

    def delete(self, id: str) -> bool:
        try:
            self._collection.delete(ids=[id])
            return True
        except Exception as e:
            log.error("delete error: %s", e, exc_info=True)
            return False

    def delete_many(self, ids: list[str]) -> bool:
        """Batch delete — used by episodic memory's decay lifecycle to
        remove all source rows of a merge in one call rather than one
        delete() per row."""
        if not ids:
            return True
        try:
            self._collection.delete(ids=ids)
            return True
        except Exception as e:
            log.error("delete_many error: %s", e, exc_info=True)
            return False

    def count(self) -> int:
        try:
            return self._collection.count()
        except Exception:
            return 0

    # ------------------- helpers functions ----------------------------#

    def find_duplicate(self, text: str, threshold: float = 0.08) -> dict | None:
        """
            Returns the closest existing memory if cosine distance <= threshold.
            Call this before add() to avoid storing near-duplicate facts.
            threshold=0.08  ≈ 92%+ similarity.
        """
        if self._collection.count() == 0:
            return None
        results = self.search(text, k=1)
        if results and results[0]["score"] <= threshold:
            return results[0]
        return None

    @staticmethod
    def _sanitise(metadata: dict[str, Any]) -> dict[str, Any]:
        """ChromaDB only accepts str | int | float | bool. Convert everything else."""
        out = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                out[k] = v
            elif v is None:
                out[k] = ""
            else:
                out[k] = str(v)
        return out