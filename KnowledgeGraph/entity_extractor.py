"""
KnowledgeGraph/entity_extractor.py

STEP 1 of the sleep pipeline: extract entities and candidate relations
from a SessionBundle using one background LLM call.

POSITION IN PIPELINE:
  memory_selector  →  [entity_extractor]  →  entity_resolver
                              ↓
                       ExtractionResult

UNIT OF WORK: SessionBundle (one complete session)
  The previous design took list[MemoryRecord] — individual ChromaDB
  memories with no session context. This version takes one SessionBundle
  per call, which contains:
    conversation_text  — chunk summaries (richest extraction signal)
    episode_text       — title + summary from episodic ChromaDB
    semantic_texts     — individual facts from semantic ChromaDB

  ONE LLM CALL PER BUNDLE. Sessions are self-contained context units.
  Mixing sessions in one prompt would cause the model to conflate
  entities across sessions. sleep_scheduler calls this once per bundle.

RETURN CONVENTIONS:
  ExtractionResult  — success. entities list may be empty if the LLM
                      found nothing worth extracting.
  None              — LLM call failed. Session stays pending in
                      kg_sleep_queue for retry on the next /sleep.

  None vs ExtractionResult(entities=[]) matters:
    None                  → retry later, do not mark processed.
    ExtractionResult([])  → nothing found, mark session processed
                            so it is not re-examined every /sleep.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

import LLMEngine.llm_request_lock as llm_request_lock
from GlobalHelpers.config import settings
from GlobalHelpers.logger import get_logger
from KnowledgeGraph.constants import (
    NODE_TYPES,
    RELATION_TYPES,
    MIN_ENTITY_CONFIDENCE,
    ENTITY_EXTRACTION_TIMEOUT,
    ENTITY_EXTRACTION_MAX_TOKENS,
    PIPELINE_TEMPERATURE,
    build_entity_extraction_system,
    build_entity_extraction_user_from_bundle,
)
from KnowledgeGraph.memory_selector import SessionBundle

log = get_logger(__name__)

# Built once at module load — vocabulary does not change between calls
_SYSTEM_PROMPT: str = build_entity_extraction_system()


# ---------------------------------------------------------------------------
# Output data structures
# ---------------------------------------------------------------------------

@dataclass
class ExtractedEntity:
    """
    A single entity extracted from a session.

    name:       Canonical name. entity_resolver normalises case and
                matches against the graph.
    type:       One of NODE_TYPES. Defaults to "Concept" if the LLM
                returns an unrecognised type.
    heading:    4-5 word description. Used for keyword indexing.
    aliases:    Alternative names seen in the session texts. Passed to
                entity_resolver to check the alias index before creating
                a new node.
    confidence: 0.0-1.0. entity_resolver filters on MIN_ENTITY_CONFIDENCE.
    """
    name:       str
    type:       str
    heading:    str
    aliases:    list[str] = field(default_factory=list)
    confidence: float     = 0.5


@dataclass
class CandidateRelation:
    """
    A proposed relationship between two entities in the session.

    source / target: entity names exactly as they appear in the
                     ExtractedEntity list. entity_resolver maps these
                     to node ids.
    relation:        One of RELATION_TYPES. Dropped if invalid.
    confidence:      0.0-1.0.
    reasoning:       One sentence. Written to kg_graph_logs for audit.
    """
    source:     str
    target:     str
    relation:   str
    confidence: float = 0.5
    reasoning:  str   = ""


@dataclass
class ExtractionResult:
    """
    Output of one entity extraction call — one per SessionBundle.

    entities:            Extracted entities. May be empty.
    candidate_relations: Proposed relationships. May be empty.
    session_id:          The session this result came from.
                         sleep_scheduler uses this to call
                         mark_processed(session_id) on success.
    memory_ids:          [episodic_memory_id] + semantic_memory_ids
                         from the bundle. Passed through so
                         graph_updater can write kg_memory_nodes links.
    raw_response:        Raw LLM output string. Kept for debugging.
    """
    entities:            list[ExtractedEntity]   = field(default_factory=list)
    candidate_relations: list[CandidateRelation] = field(default_factory=list)
    session_id:          str                     = ""
    memory_ids:          list[str]               = field(default_factory=list)
    raw_response:        str                     = ""


# ---------------------------------------------------------------------------
# Internal: parse and validate LLM output
# ---------------------------------------------------------------------------

def _strip_fences(raw: str) -> str:
    """Remove accidental markdown code fences from LLM output."""
    if raw.startswith("```"):
        return "\n".join(
            l for l in raw.splitlines()
            if not l.strip().startswith("```")
        ).strip()
    return raw


def _parse_entities(raw_entities: list) -> list[ExtractedEntity]:
    """
    Parse and validate the 'entities' list from the LLM JSON.

    Normalisation rules:
      name        — stripped. All-lowercase names are title-cased.
      type        — must be in NODE_TYPES, defaults to "Concept".
      aliases     — lowercased, deduplicated, empty strings removed.
      confidence  — clamped to [0.0, 1.0].

    Entities with an empty name or duplicate names (case-insensitive)
    are dropped. Entities below MIN_ENTITY_CONFIDENCE are included —
    entity_resolver applies the threshold.
    """
    result: list[ExtractedEntity] = []
    seen:   set[str]              = set()

    for raw in raw_entities:
        if not isinstance(raw, dict):
            continue

        name = str(raw.get("name") or "").strip()
        if not name:
            continue
        if name == name.lower():
            name = name.title()

        name_lower = name.lower()
        if name_lower in seen:
            log.debug("_parse_entities: duplicate name=%r — skipped.", name)
            continue
        seen.add(name_lower)

        raw_type = str(raw.get("type") or "").strip()
        if raw_type not in NODE_TYPES:
            if raw_type:
                log.warning(
                    "_parse_entities: unknown type=%r for %r — defaulting to Concept.",
                    raw_type, name,
                )
            entity_type = "Concept"
        else:
            entity_type = raw_type

        heading = str(raw.get("heading") or "").strip()

        raw_aliases = raw.get("aliases") or []
        aliases: list[str] = []
        seen_a: set[str]   = set()
        if isinstance(raw_aliases, list):
            for a in raw_aliases:
                a_str = str(a).strip().lower()
                if a_str and a_str not in seen_a and a_str != name_lower:
                    aliases.append(a_str)
                    seen_a.add(a_str)

        try:
            confidence = float(max(0.0, min(1.0, raw.get("confidence", 0.5))))
        except (TypeError, ValueError):
            confidence = 0.5

        if confidence < MIN_ENTITY_CONFIDENCE:
            log.debug(
                "_parse_entities: %r conf=%.2f below MIN — included, "
                "entity_resolver will filter.",
                name, confidence,
            )

        result.append(ExtractedEntity(
            name=name, type=entity_type, heading=heading,
            aliases=aliases, confidence=confidence,
        ))

    return result


def _parse_candidate_relations(
    raw_relations:     list,
    valid_entity_names: set[str],
) -> list[CandidateRelation]:
    """
    Parse and validate the 'candidate_relations' list from the LLM JSON.

    Validation rules:
      - source and target must both appear in valid_entity_names
        (case-insensitive). Relations referencing entities not in the
        extracted list are hallucination — dropped.
      - relation must be in RELATION_TYPES — dropped if not.
      - source != target — self-relations dropped.
      - confidence clamped to [0.0, 1.0].
    """
    result:      list[CandidateRelation] = []
    valid_lower: set[str]               = {n.lower() for n in valid_entity_names}

    for raw in raw_relations:
        if not isinstance(raw, dict):
            continue

        source   = str(raw.get("source")   or "").strip()
        target   = str(raw.get("target")   or "").strip()
        relation = str(raw.get("relation") or "").strip()

        if not source or not target or not relation:
            continue
        if source.lower() == target.lower():
            continue
        if source.lower() not in valid_lower:
            log.debug("_parse_candidate_relations: source=%r not in entities — dropped.", source)
            continue
        if target.lower() not in valid_lower:
            log.debug("_parse_candidate_relations: target=%r not in entities — dropped.", target)
            continue
        if relation not in RELATION_TYPES:
            log.warning("_parse_candidate_relations: relation=%r not in RELATION_TYPES — dropped.", relation)
            continue

        try:
            confidence = float(max(0.0, min(1.0, raw.get("confidence", 0.5))))
        except (TypeError, ValueError):
            confidence = 0.5

        result.append(CandidateRelation(
            source    = source,
            target    = target,
            relation  = relation,
            confidence = confidence,
            reasoning  = str(raw.get("reasoning") or "").strip(),
        ))

    return result


def _parse_response(
    raw:        str,
    session_id: str,
    memory_ids: list[str],
) -> ExtractionResult:
    """
    Parse raw LLM JSON into an ExtractionResult.

    Returns ExtractionResult with empty lists on any parse failure —
    never raises. Distinguishable from None (LLM call failure):
      ExtractionResult([]) → LLM ran, found nothing → mark processed.
      None                 → LLM call failed → retry later.
    """
    raw = _strip_fences(raw)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        log.warning(
            "_parse_response: JSON decode failed for session=%s: %s\nraw: %s",
            session_id, e, raw[:300],
        )
        repaired = _try_repair_json(raw)
        if repaired is not None:
            log.info(
                "_parse_response: JSON repair succeeded for session=%s.",
                session_id,
            )
            parsed = repaired
        else:
            log.warning(
                "_parse_response: JSON repair failed for session=%s.",
                session_id,
            )

    if not isinstance(parsed, dict):
        log.warning(
            "_parse_response: top-level is %s for session=%s.",
            type(parsed).__name__, session_id,
        )
        return ExtractionResult(session_id=session_id, memory_ids=memory_ids, raw_response=raw)

    raw_entities  = parsed.get("entities")           or []
    raw_relations = parsed.get("candidate_relations") or []

    if not isinstance(raw_entities,  list): raw_entities  = []
    if not isinstance(raw_relations, list): raw_relations = []

    entities            = _parse_entities(raw_entities)
    valid_names         = {e.name for e in entities}
    candidate_relations = _parse_candidate_relations(raw_relations, valid_names)

    log.info(
        "_parse_response: session=%s → %d entities, %d candidate_relations.",
        session_id, len(entities), len(candidate_relations),
    )
    return ExtractionResult(
        entities            = entities,
        candidate_relations = candidate_relations,
        session_id          = session_id,
        memory_ids          = memory_ids,
        raw_response        = raw,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_entities_from_bundle(
    bundle: Optional[SessionBundle],
) -> Optional[ExtractionResult]:
    """
    Extract entities and candidate relations from one SessionBundle.

    Makes ONE background LLM call using the session-aware prompt:
    conversation narrative + episode summary + semantic facts, in that
    priority order.

    Args:
      bundle: A SessionBundle from memory_selector.get_next_batch().
              None is accepted defensively and returns None.

    Returns:
      ExtractionResult  — LLM call succeeded. entities may be empty.
      None              — LLM call or network failed. Session stays
                          pending in kg_sleep_queue for the next /sleep.
    """
    if bundle is None:
        log.warning("extract_entities_from_bundle: called with None bundle.")
        return None

    session_id = bundle.session_id
    if not session_id:
        log.warning("extract_entities_from_bundle: bundle has empty session_id.")
        return None

    # Build the memory_ids list: episodic first, then semantic
    memory_ids: list[str] = []
    if bundle.episodic_memory_id:
        memory_ids.append(bundle.episodic_memory_id)
    memory_ids.extend(bundle.semantic_memory_ids)

    user_content = build_entity_extraction_user_from_bundle(bundle)

    log.info(
        "extract_entities_from_bundle: session=%s "
        "conv=%d ep=%d sem_facts=%d.",
        session_id,
        len(bundle.conversation_text),
        len(bundle.episode_text),
        len(bundle.semantic_texts),
    )

    try:
        response = llm_request_lock.post_completion(
            {
                "model":    settings.llm_model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_content},
                ],
                "temperature":          PIPELINE_TEMPERATURE,
                "max_tokens":           ENTITY_EXTRACTION_MAX_TOKENS,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            role    = "main",
            timeout = ENTITY_EXTRACTION_TIMEOUT,
        )
        response.raise_for_status()
        raw = (
            response.json()
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
    except Exception as e:
        log.error(
            "extract_entities_from_bundle: LLM call failed for session=%s: %s",
            session_id, e, exc_info=True,
        )
        return None

    if not raw:
        log.warning(
            "extract_entities_from_bundle: empty LLM response for session=%s.",
            session_id,
        )
        return ExtractionResult(session_id=session_id, memory_ids=memory_ids, raw_response="")

    return _parse_response(raw, session_id, memory_ids)

def _try_repair_json(raw: str) -> Optional[dict]:
    """Best-effort recovery for truncated JSON: cut back to the last
    complete entity object and close open brackets."""
    import re
    # find the last complete "}" that closes an entity dict
    last_close = raw.rfind("}")
    if last_close == -1:
        return None
    candidate = raw[:last_close + 1]
    # balance brackets
    opens_sq, closes_sq = candidate.count("["), candidate.count("]")
    opens_cu, closes_cu = candidate.count("{"), candidate.count("}")
    candidate += "]" * (opens_sq - closes_sq)
    candidate += "}" * (opens_cu - closes_cu)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None