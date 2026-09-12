"""
Database/kg_constants.py

Shared constants for the Knowledge Graph DB layer.
Imported by all kg_*_client.py modules — no other KG module imports
from here except these clients, so there is no import cycle.
"""

from __future__ import annotations

from datetime import datetime, timezone
import uuid


NODE_TYPES = frozenset({
    "Person",
    "Project",
    "Technology",
    "Organization",
    "Place",
    "Event",
    "Concept",
})

RELATION_TYPES = frozenset({
    "uses",
    "built_with",
    "depends_on",
    "contains",
    "created",
    "replaced",
    "part_of",
    "related_to",
    "located_in",
    "knows",
    "worked_on",
    "prefers",
    "learned",
    "mentions",
    "contradicts",
})

_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "was", "are", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "can", "its",
    "it", "this", "that", "these", "those", "my", "your", "his", "her",
    "our", "their", "i", "we", "you", "he", "she", "they", "as", "if",
})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return str(uuid.uuid4())