"""
KnowledgeGraph/operation_proposer.py

STEP 4 of the sleep pipeline: propose concrete graph operations based on
resolved entities, their local subgraph, and the candidate relations from
entity extraction.

POSITION IN PIPELINE:
  memory_selector  →  entity_extractor  →  entity_resolver  →  subgraph_retriever
                                                                    ↓
                                                            [operation_proposer]
                                                                    ↓
                                                          list[ProposedOperation]

UNIT OF WORK: One LLM call per batch of resolved entities.
  The proposer receives:
    - Resolved entities with their graph node IDs (from entity_resolver)
    - The local subgraph around those nodes (from subgraph_retriever)
    - Candidate relations from the initial extraction (from entity_extractor)
    - Session memory texts for evidence

  It makes ONE background LLM call that proposes the minimum set of graph
  operations (insert_edge, update_edge_confidence, deactivate_edge, add_alias)
  to accurately capture the relationships in the memories.

OPERATION TYPES:
  - insert_edge:              Create a new directed relationship between nodes.
  - update_edge_confidence:   Adjust confidence of an existing edge.
  - deactivate_edge:          Mark an edge as no longer current.
  - add_alias:                Add an alternative name to an existing node.

  Node creation is NOT here — entity_resolver handles that. This module
  only proposes edge-level and alias operations.

CANDIDATE RELATIONS:
  The entity_extractor extracts candidate_relations as hints. The proposer
  uses these to guide the LLM but does NOT blindly apply them — the LLM
  sees the subgraph and decides whether each candidate is already covered
  by an existing edge, needs updating, or should be inserted fresh.

RETURN CONVENTIONS:
  list[ProposedOperation]  — LLM call succeeded. May be empty if the
                             graph already reflects the memories.
  None                     — LLM call failed. Session stays pending in
                             kg_sleep_queue for retry on the next /sleep.
"""

from __future__ import annotations

import json
import requests
from dataclasses import dataclass, field
from typing import Optional

import LLMEngine.llm_request_lock as llm_request_lock
from GlobalHelpers.config import settings
from GlobalHelpers.logger import get_logger
from KnowledgeGraph.constants import (
    VALID_OPERATION_TYPES,
    OPERATION_REQUIRED_FIELDS,
    CONFIDENCE_MIN,
    CONFIDENCE_MAX,
    OPERATION_PROPOSAL_TIMEOUT,
    OPERATION_PROPOSAL_MAX_TOKENS,
    PIPELINE_TEMPERATURE,
    build_operation_proposal_system,
    build_operation_proposal_user,
)
from KnowledgeGraph.entity_resolver import ResolutionResult
from KnowledgeGraph.memory_selector import SessionBundle
from KnowledgeGraph.subgraph_retriever import Subgraph

log = get_logger(__name__)

# Built once at module load — vocabulary does not change between calls
_SYSTEM_PROMPT = build_operation_proposal_system()

# Retry count for operation_proposer LLM response timeout
retries = 0


# Output data structure


@dataclass
class ProposedOperation:
    """
    A single graph operation proposed by the LLM.

    op_type:              One of VALID_OPERATION_TYPES.
    source_id / target_id: Node ids for insert_edge.
    relation:             Relation type for insert_edge.
    evidence_memory_ids:  Memory ids supporting this operation (for audit trail).
    confidence:           0.0-1.0 for insert_edge and update_edge_confidence.
    edge_id:              Edge id for update_edge_confidence and deactivate_edge.
    reasoning:            Explanation for deactivate_edge.
    node_id / alias:      For add_alias operations.
    raw:                  The original LLM JSON dict, kept for debugging.
    """
    op_type: str
    source_id: str = ""
    target_id: str = ""
    relation: str = ""
    evidence_memory_ids: list = field(default_factory=list)
    confidence: float = 0.0
    edge_id: str = ""
    reasoning: str = ""
    node_id: str = ""
    alias: str = ""
    raw: dict = field(default_factory=dict)


# Internal: parse and validate LLM output


def _strip_fences(raw):
    """Remove accidental markdown code fences from LLM output."""
    if raw.startswith("```"):
        return "\n".join(
            l for l in raw.splitlines()
            if not l.strip().startswith("```")
        ).strip()
    return raw


def _parse_operation(raw, valid_ids: set[str]):
    """
    Parse a single operation dict from the LLM JSON into a ProposedOperation.

    Validates:
      - op_type is in VALID_OPERATION_TYPES
      - Required fields for the op_type are present and non-empty
      - For insert_edge: source_id and target_id exist in valid_ids
      - Confidence is clamped to [CONFIDENCE_MIN, CONFIDENCE_MAX]

    Returns None if validation fails for any reason.
    """
    if not isinstance(raw, dict):
        return None

    op_type = str(raw.get("type") or "").strip()
    if not op_type or op_type not in VALID_OPERATION_TYPES:
        return None

    # Check required fields
    for f in OPERATION_REQUIRED_FIELDS.get(op_type, []):
        val = raw.get(f)
        if val is None or (isinstance(val, (str, list)) and not val):
            return None

    op = ProposedOperation(op_type=op_type, raw=raw)

    try:
        if op_type == "insert_edge":
            op.source_id = str(raw["source_id"]).strip()
            op.target_id = str(raw["target_id"]).strip()
            if op.source_id not in valid_ids or op.target_id not in valid_ids:
                log.warning(
                    "propose_operations: hallucinated id — source=%r target=%r not in resolved set.",
                    op.source_id, op.target_id,
                )
                return None
            op.relation = str(raw["relation"]).strip()
            op.confidence = float(max(CONFIDENCE_MIN, min(CONFIDENCE_MAX, raw.get("confidence", 0.5))))
            ev = raw.get("evidence_memory_ids") or []
            op.evidence_memory_ids = [str(m) for m in ev if m] if isinstance(ev, list) else []

        elif op_type == "update_edge_confidence":
            op.edge_id = str(raw["edge_id"]).strip()
            op.confidence = float(max(CONFIDENCE_MIN, min(CONFIDENCE_MAX, raw.get("confidence", 0.5))))

        elif op_type == "deactivate_edge":
            op.edge_id = str(raw["edge_id"]).strip()
            op.reasoning = str(raw.get("reasoning") or "").strip()

        elif op_type == "add_alias":
            op.node_id = str(raw["node_id"]).strip()
            op.alias = str(raw["alias"]).strip().lower()

    except Exception:
        return None

    return op


def _parse_response(raw, bundle, valid_ids: set[str]):
    """
    Parse the full LLM response into a list of ProposedOperations.

    Strips markdown fences, parses JSON, validates each operation,
    and filters out any that fail validation. Returns [] if the LLM
    proposed no operations (correct output — graph already reflects memories).
    Returns None if the response is unparseable.
    """
    raw = _strip_fences(raw)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        log.warning(
            "propose_operations: JSON decode error: %s. Raw: %r", e, raw,
        )
        return None

    if not isinstance(parsed, dict):
        return None

    raw_ops = parsed.get("operations")
    if raw_ops is None or not isinstance(raw_ops, list):
        return None
    if not raw_ops:
        return []

    return [
        op for op in (_parse_operation(r, valid_ids) for r in raw_ops)
        if op is not None
    ]


# Public API
def propose_operations(
    resolved: list[ResolutionResult],
    subgraph: Subgraph,
    bundles: list[SessionBundle],
    candidate_relations=None,
):
    """
    Propose graph operations for a batch of resolved entities.

    Makes ONE background LLM call using the session-aware prompt:
    resolved entities with node IDs, the local subgraph context, memory
    texts for evidence, and candidate relation hints from extraction.

    Args:
      resolved:             list[ResolutionResult] from entity_resolver.
      subgraph:             Subgraph from subgraph_retriever.fetch_subgraph().
      bundles:              list[SessionBundle] — uses first bundle for
                            memory context.
      candidate_relations:  Optional list[CandidateRelation] from extraction.

    Returns:
      list[ProposedOperation]  — LLM call succeeded. May be empty.
      None                     — LLM call failed. Session stays pending.
    """
    valid = [r for r in resolved if r.node_id]
    if not valid:
        return []

    # Build the set of valid node ids for hallucination detection
    valid_ids = {r.node_id for r in valid}

    # Map entity names to node ids for candidate relation hint resolution
    name_to_id = {r.node_name.lower(): r.node_id for r in valid}
    hints = []
    for cr in (candidate_relations or []):
        sid = name_to_id.get(cr.source.lower())
        tid = name_to_id.get(cr.target.lower())
        if sid and tid:
            hints.append(
                f"  {cr.source} --{cr.relation}--> {cr.target}  "
                f"(conf={cr.confidence:.2f}: {cr.reasoning})"
            )

    # Build prompt inputs
    resolved_for_prompt = [
        {
            "name": r.node_name,
            "node_id": r.node_id,
            "type": r.entity.type,
            "heading": r.entity.heading,
            "is_new": r.is_new,
        }
        for r in valid
    ]

    # Use first bundle for memory context
    bundle = bundles[0] if bundles else None
    mem_ids_texts = []
    if bundle:
        if bundle.episodic_memory_id:
            mem_ids_texts.append((
                bundle.episodic_memory_id,
                bundle.episode_text or bundle.conversation_text[:200],
            ))
        for mid, text in zip(bundle.semantic_memory_ids, bundle.semantic_texts):
            mem_ids_texts.append((mid, text[:200]))

    user_content = build_operation_proposal_user(
        resolved_for_prompt, subgraph.text, mem_ids_texts, hints,
    )

    try:
        response = llm_request_lock.post_completion(
            {
                "model": settings.llm_model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_content},
                ],
                "temperature": PIPELINE_TEMPERATURE,
                "max_tokens": OPERATION_PROPOSAL_MAX_TOKENS,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            role="main",
            timeout=OPERATION_PROPOSAL_TIMEOUT,
        )
        response.raise_for_status()
        raw = (
            response.json()
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
    except requests.exceptions.Timeout as e:
        retries += 1
        log.warning(
            "propose_operations: LLM call timed out, retry count = %s", retries,
        )
    except Exception as e:
        log.error(
            "propose_operations: LLM call failed: %s", e, exc_info=True,
        )
        return None

    if not raw:
        return None

    return _parse_response(raw, bundle, valid_ids)
