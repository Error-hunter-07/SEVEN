"""
KnowledgeGraph/entity_resolver.py

STEP 2 of the sleep pipeline: resolve extracted entities against the
existing Knowledge Graph and create new nodes when no match is found.

POSITION IN PIPELINE:
  memory_selector  →  entity_extractor  →  [entity_resolver]  →  subgraph_retriever
                                  ↓
                           list[ResolutionResult]

UNIT OF WORK: ExtractionResult (one per SessionBundle)
  Receives all entities extracted from a single session by entity_extractor.
  Each entity is resolved independently against the graph using a cascade
  of matching strategies. The output — list[ResolutionResult] — carries
  enough metadata for the operation proposer and for sleep_scheduler's
  logging.

MATCHING CASCADE (in order):
  1. Exact name match     — entity.name matches a node name (case-insensitive)
  2. Self-alias match     — entity.name matches one of a node's aliases
  3. Alias-list match     — one of entity.aliases matches a node's alias
  4. Keyword match        — Jaccard similarity of extracted keywords exceeds
                            MERGE_CONFIDENCE_THRESHOLD
  5. Create new node      — no match found; insert a new node into the graph

Prefix matching was deliberately removed: it merged dissimilar entities
(e.g. "Java" ↔ "JavaScript", "React" ↔ "Reactive"). The keyword cascade
is conservative enough to avoid these false positives.

SIDE EFFECTS:
  - Aliases from the extracted entity are added to the matched node
    (via kg.add_alias).
  - If the entity's confidence is higher than the node's current confidence,
    the node's confidence is updated (via kg.update_node).
  - New nodes are inserted with importance = min(entity.confidence, 0.6)
    to avoid a single extraction inflating a node's importance too early.

RETURN CONVENTIONS:
  ResolutionResult with node_id=""  — entity was skipped (below confidence
                                      threshold or DB write failed).
  ResolutionResult with is_new=True — node was just created by this call.
  ResolutionResult with is_new=False — existing node matched and confirmed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import Database.kg_db_client as kg
from GlobalHelpers.logger import get_logger
from KnowledgeGraph.constants import MIN_ENTITY_CONFIDENCE, MERGE_CONFIDENCE_THRESHOLD
from KnowledgeGraph.entity_extractor import ExtractedEntity, ExtractionResult

log = get_logger(__name__)


# Output data structure

@dataclass
class ResolutionResult:
    """
    Result of resolving one ExtractedEntity against the Knowledge Graph.

    entity:           The original extracted entity.
    node_id:          Graph node id this entity maps to. Empty string if
                      the entity was skipped or creation failed.
    is_new:           True if this node was created during this call.
    match_method:     How the match was found: "exact_name", "alias_self",
                      "alias_list", "keyword", "created", or "skipped".
    match_confidence: Confidence score for the match (1.0 for exact/alias,
                      Jaccard score for keyword, entity.confidence for created).
    node_name:        The canonical name of the matched/created node.
    """
    entity:           ExtractedEntity
    node_id:          str
    is_new:           bool
    match_method:     str
    match_confidence: float
    node_name:        str


# Internal: keyword-based matching
def _keyword_score(entity, node):
    """
    Compute Jaccard similarity between the entity's extracted keywords
    and the node's stored keywords. Returns 0.0 if either set is empty.
    """
    ek = set(kg._extract_keywords(entity.name, entity.heading))
    if not ek:
        return 0.0
    nk = set(kg.get_keywords_for_node(node["id"]))
    if not nk:
        return 0.0
    inter = ek & nk
    union = ek | nk
    return len(inter) / len(union) if union else 0.0


def _best_keyword_match(entity):
    """
    Find the best keyword-matching node for an extracted entity.

    Searches the graph's keyword index for each keyword in the entity,
    collects candidate nodes, and returns the one with the highest
    Jaccard score — provided it meets MERGE_CONFIDENCE_THRESHOLD.

    Returns (node, score) or None if no match is close enough.
    """
    keywords = kg._extract_keywords(entity.name, entity.heading)
    if not keywords:
        return None

    candidates = {}
    for kw in keywords:
        for node in kg.get_nodes_by_keyword(kw, limit=10):
            if node["id"] not in candidates:
                candidates[node["id"]] = node

    if not candidates:
        return None

    best_node, best_score = None, 0.0
    for node in candidates.values():
        score = _keyword_score(entity, node)
        if score > best_score:
            best_score = score
            best_node = node

    if best_node is None or best_score < MERGE_CONFIDENCE_THRESHOLD:
        return None
    return (best_node, best_score)


# Internal: side effects on matched nodes

def _apply_side_effects(entity, node, is_new):
    """
    Apply alias additions and confidence updates to a matched/created node.

    For existing nodes: adds all entity aliases, updates confidence if
    the entity's confidence is higher.
    For new nodes: aliases are already added during insert_node.
    """
    for alias in entity.aliases:
        if alias:
            kg.add_alias(node["id"], alias)
    if not is_new and entity.confidence > node["confidence"]:
        kg.update_node(node["id"], confidence=entity.confidence)


# Internal: resolve a single entity

def _resolve_one(entity):
    """
    Resolve a single ExtractedEntity against the graph using the matching
    cascade. Returns a ResolutionResult with the match outcome.

    The cascade tries exact name → self-alias → alias-list → keyword
    match before falling back to creating a new node. Each step is
    attempted only if all previous steps failed.
    """
    # Below confidence threshold — skip entirely
    if entity.confidence < MIN_ENTITY_CONFIDENCE:
        return ResolutionResult(
            entity=entity, node_id="", is_new=False,
            match_method="skipped", match_confidence=0.0,
            node_name=entity.name,
        )

    # 1. Exact name match
    node = kg.get_node_by_name(entity.name)
    if node:
        _apply_side_effects(entity, node, False)
        return ResolutionResult(
            entity=entity, node_id=node["id"], is_new=False,
            match_method="exact_name", match_confidence=1.0,
            node_name=node["name"],
        )

    # 2. Self-alias match (entity name is already an alias of a node)
    node = kg.get_node_by_alias(entity.name)
    if node:
        _apply_side_effects(entity, node, False)
        return ResolutionResult(
            entity=entity, node_id=node["id"], is_new=False,
            match_method="alias_self", match_confidence=1.0,
            node_name=node["name"],
        )

    # 3. Alias-list match (one of entity's aliases matches a node's alias)
    for alias in entity.aliases:
        node = kg.get_node_by_alias(alias)
        if node:
            _apply_side_effects(entity, node, False)
            return ResolutionResult(
                entity=entity, node_id=node["id"], is_new=False,
                match_method="alias_list", match_confidence=1.0,
                node_name=node["name"],
            )

    # 4. Keyword match (Jaccard similarity above threshold)
    kw_match = _best_keyword_match(entity)
    if kw_match:
        node, score = kw_match
        _apply_side_effects(entity, node, False)
        kg.add_alias(node["id"], entity.name)
        return ResolutionResult(
            entity=entity, node_id=node["id"], is_new=False,
            match_method="keyword", match_confidence=score,
            node_name=node["name"],
        )

    # 5. No match — create a new node
    new_id = kg.insert_node(
        name=entity.name,
        type=entity.type,
        heading=entity.heading,
        confidence=entity.confidence,
        importance=min(entity.confidence, 0.6),
    )
    if not new_id:
        # Race condition: another process may have created the node
        existing = kg.get_node_by_name(entity.name)
        if existing:
            _apply_side_effects(entity, existing, False)
            return ResolutionResult(
                entity=entity, node_id=existing["id"], is_new=False,
                match_method="exact_name", match_confidence=1.0,
                node_name=existing["name"],
            )
        return ResolutionResult(
            entity=entity, node_id="", is_new=False,
            match_method="skipped", match_confidence=0.0,
            node_name=entity.name,
        )

    new_node = kg.get_node_by_id(new_id) or {
        "id": new_id, "name": entity.name, "confidence": entity.confidence,
    }
    _apply_side_effects(entity, new_node, True)
    return ResolutionResult(
        entity=entity, node_id=new_id, is_new=True,
        match_method="created", match_confidence=entity.confidence,
        node_name=entity.name,
    )


# Public API

def resolve_entities(extraction: ExtractionResult) -> list[ResolutionResult]:
    """
    Resolve all entities from an ExtractionResult against the Knowledge Graph.

    Each entity is resolved independently. Failures on individual entities
    are caught and logged — they produce a ResolutionResult with
    node_id="" (skipped) rather than failing the entire batch.

    Args:
      extraction: Output from entity_extractor.extract_entities_from_bundle().

    Returns:
      list[ResolutionResult] — one per extracted entity, in the same order.
      Empty list if extraction has no entities.
    """
    if not extraction.entities:
        return []

    results = []
    for entity in extraction.entities:
        try:
            results.append(_resolve_one(entity))
        except Exception:
            log.exception(
                "resolve_entities: error on entity=%r", entity.name,
            )
            results.append(ResolutionResult(
                entity=entity, node_id="", is_new=False,
                match_method="skipped", match_confidence=0.0,
                node_name=entity.name,
            ))
    return results
