from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import Database.kg_db_client as kg
from GlobalHelpers.logger import get_logger
from KnowledgeGraph.constants import MIN_ENTITY_CONFIDENCE, MERGE_CONFIDENCE_THRESHOLD
from KnowledgeGraph.entity_extractor import ExtractedEntity, ExtractionResult
log = get_logger(__name__)

@dataclass
class ResolutionResult:
    entity:           ExtractedEntity
    node_id:          str
    is_new:           bool
    match_method:     str
    match_confidence: float
    node_name:        str

def _keyword_score(entity, node):
    ek = set(kg._extract_keywords(entity.name, entity.heading))
    if not ek: return 0.0
    nk = set(kg.get_keywords_for_node(node["id"]))
    if not nk: return 0.0
    inter = ek & nk; union = ek | nk
    return len(inter)/len(union) if union else 0.0

def _best_keyword_match(entity):
    keywords = kg._extract_keywords(entity.name, entity.heading)
    if not keywords: return None
    candidates = {}
    for kw in keywords:
        for node in kg.get_nodes_by_keyword(kw, limit=10):
            if node["id"] not in candidates: candidates[node["id"]] = node
    if not candidates: return None
    best_node, best_score = None, 0.0
    for node in candidates.values():
        score = _keyword_score(entity, node)
        if score > best_score: best_score = score; best_node = node
    if best_node is None or best_score < MERGE_CONFIDENCE_THRESHOLD: return None
    return (best_node, best_score)

#The following block is commented because the prefix match logic over here can merge entities which are not even similar like java vs javascript, react vs reactive, etc

# def _best_prefix_match(entity):
#     prefix = entity.name[:5].strip()
#     if len(prefix) < 3: return None
#     candidates = kg.search_nodes_by_name_prefix(prefix, limit=5)
#     if not candidates: return None
#     ek = set(kg._extract_keywords(entity.name, entity.heading))
#     for node in candidates:
#         if node["importance"] < 0.6: continue
#         nk = set(kg.get_keywords_for_node(node["id"]))
#         if ek & nk: return node
#     return None

def _apply_side_effects(entity, node, is_new):
    for alias in entity.aliases:
        if alias: kg.add_alias(node["id"], alias)
    if not is_new and entity.confidence > node["confidence"]:
        kg.update_node(node["id"], confidence=entity.confidence)

def _resolve_one(entity):
    if entity.confidence < MIN_ENTITY_CONFIDENCE:
        return ResolutionResult(entity=entity, node_id="", is_new=False, match_method="skipped", match_confidence=0.0, node_name=entity.name)
    node = kg.get_node_by_name(entity.name)
    if node:
        _apply_side_effects(entity, node, False)
        return ResolutionResult(entity=entity, node_id=node["id"], is_new=False, match_method="exact_name", match_confidence=1.0, node_name=node["name"])
    node = kg.get_node_by_alias(entity.name)
    if node:
        _apply_side_effects(entity, node, False)
        return ResolutionResult(entity=entity, node_id=node["id"], is_new=False, match_method="alias_self", match_confidence=1.0, node_name=node["name"])
    for alias in entity.aliases:
        node = kg.get_node_by_alias(alias)
        if node:
            _apply_side_effects(entity, node, False)
            return ResolutionResult(entity=entity, node_id=node["id"], is_new=False, match_method="alias_list", match_confidence=1.0, node_name=node["name"])
    kw_match = _best_keyword_match(entity)
    if kw_match:
        node, score = kw_match
        _apply_side_effects(entity, node, False)
        kg.add_alias(node["id"], entity.name)
        return ResolutionResult(entity=entity, node_id=node["id"], is_new=False, match_method="keyword", match_confidence=score, node_name=node["name"])
    
    #Removed from the prefix match logic from here too

    # prefix_match = _best_prefix_match(entity)
    # if prefix_match:
    #     node = prefix_match
    #     _apply_side_effects(entity, node, False)
    #     kg.add_alias(node["id"], entity.name)
    #     return ResolutionResult(entity=entity, node_id=node["id"], is_new=False, match_method="prefix", match_confidence=0.7, node_name=node["name"])
    new_id = kg.insert_node(name=entity.name, type=entity.type, heading=entity.heading, confidence=entity.confidence, importance=min(entity.confidence, 0.6))
    if not new_id:
        existing = kg.get_node_by_name(entity.name)
        if existing:
            _apply_side_effects(entity, existing, False)
            return ResolutionResult(entity=entity, node_id=existing["id"], is_new=False, match_method="exact_name", match_confidence=1.0, node_name=existing["name"])
        return ResolutionResult(entity=entity, node_id="", is_new=False, match_method="skipped", match_confidence=0.0, node_name=entity.name)
    new_node = kg.get_node_by_id(new_id) or {"id": new_id, "name": entity.name, "confidence": entity.confidence}
    _apply_side_effects(entity, new_node, True)
    return ResolutionResult(entity=entity, node_id=new_id, is_new=True, match_method="created", match_confidence=entity.confidence, node_name=entity.name)

def resolve_entities(extraction):
    if not extraction.entities: return []
    results = []
    for entity in extraction.entities:
        try: results.append(_resolve_one(entity))
        except Exception:
            log.exception("resolve_entities: error on entity=%r", entity.name)
            results.append(ResolutionResult(entity=entity, node_id="", is_new=False, match_method="skipped", match_confidence=0.0, node_name=entity.name))
    return results