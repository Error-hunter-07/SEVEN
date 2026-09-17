"""
KnowledgeGraph/validator.py

STEP 5 of the sleep pipeline: validate every proposed operation before
it is written to the Knowledge Graph database.

POSITION IN PIPELINE:
  memory_selector  →  entity_extractor  →  entity_resolver  →  subgraph_retriever
      →  operation_proposer  →  [validator]  →  sleep_scheduler._execute_valid_ops

UNIT OF WORK: list[ProposedOperation] (from operation_proposer)
  Each operation is validated independently. Validation is purely
  deterministic — no LLM calls, no network requests. It checks
  structural correctness against the graph's current state.

WHAT VALIDATION COVERS:
  - insert_edge:            source/target nodes exist, no self-loop,
                            relation is valid, confidence meets threshold,
                            no duplicate active edge already exists.
  - update_edge_confidence: edge exists and is active.
  - deactivate_edge:        edge exists and is currently active.
  - add_alias:              node exists and alias is non-empty.

  Symmetric relations (related_to, knows, contradicts) are normalized:
  the validator swaps source_id and target_id so the lower alphabetical
  name is always the source. This prevents duplicate edges in both
  directions.

DUPLICATE EDGE HANDLING:
  When insert_edge is rejected because an active edge with the same
  relation already exists between those nodes, the validator sets
  duplicate_edge_id on the ValidationResult. sleep_scheduler auto-converts
  these rejections to update_edge_confidence using the existing edge id —
  the information is not lost.

CONFIDENCE CLAMPING:
  Values outside [CONFIDENCE_MIN, CONFIDENCE_MAX] are clamped, not rejected.
  An LLM output of 1.02 is almost certainly a rounding artefact, not a
  fundamentally bad operation.

RETURN CONVENTIONS:
  ValidationResult with is_valid=True   — operation is safe to execute.
  ValidationResult with is_valid=False  — operation rejected, reason logged.
  Duplicate-edge rejections carry duplicate_edge_id for auto-conversion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import Database.kg_db_client as kg
from GlobalHelpers.logger import get_logger
from KnowledgeGraph.constants import (
    RELATION_TYPES,
    SYMMETRIC_RELATIONS,
    MIN_EDGE_CONFIDENCE,
    CONFIDENCE_MIN,
    CONFIDENCE_MAX,
)
from KnowledgeGraph.operation_proposer import ProposedOperation

log = get_logger(__name__)

# Output data structure

@dataclass
class ValidationResult:
    """
    Result of validating one ProposedOperation.

    op:                The original proposed operation.
    is_valid:          True if the operation passed all checks.
    rejection_reason:  Human-readable explanation if is_valid is False.
    duplicate_edge_id: If rejection is due to an existing active edge,
                       this is that edge's id. sleep_scheduler uses it
                       to auto-convert to update_edge_confidence.
    """
    op: ProposedOperation
    is_valid: bool
    rejection_reason: str = ""
    duplicate_edge_id: str = ""

# Internal: per-operation-type validators

def _validate_insert_edge(op):
    """
    Validate an insert_edge operation.

    Checks:
      1. source_id node exists in the graph
      2. target_id node exists in the graph
      3. No self-loop (source_id != target_id)
      4. relation is in RELATION_TYPES
      5. confidence >= MIN_EDGE_CONFIDENCE
      6. No active edge with same relation already exists between these nodes
      7. Symmetric relations are normalized (lower name → source)

    Returns ValidationResult with is_valid=True if all checks pass.
    """
    # Node existence checks
    src = kg.get_node_by_id(op.source_id)
    if not src:
        return ValidationResult(
            op=op, is_valid=False,
            rejection_reason=f"source_id={op.source_id!r} not found.",
        )
    tgt = kg.get_node_by_id(op.target_id)
    if not tgt:
        return ValidationResult(
            op=op, is_valid=False,
            rejection_reason=f"target_id={op.target_id!r} not found.",
        )

    # Self-loop rejection
    if op.source_id == op.target_id:
        return ValidationResult(
            op=op, is_valid=False,
            rejection_reason="Self-loop rejected.",
        )

    # Relation type validation
    if op.relation not in RELATION_TYPES:
        return ValidationResult(
            op=op, is_valid=False,
            rejection_reason=f"relation={op.relation!r} not in RELATION_TYPES.",
        )

    # Confidence threshold
    if op.confidence < MIN_EDGE_CONFIDENCE:
        return ValidationResult(
            op=op, is_valid=False,
            rejection_reason=f"confidence={op.confidence:.3f} below MIN={MIN_EDGE_CONFIDENCE}.",
        )

    # Symmetric relation normalization: lower name alphabetically → source
    if op.relation in SYMMETRIC_RELATIONS and op.source_id > op.target_id:
        op.source_id, op.target_id = op.target_id, op.source_id

    # Duplicate edge check
    existing = kg.get_edge_between(
        op.source_id, op.target_id,
        relation=op.relation, active_only=True,
    )
    if existing:
        return ValidationResult(
            op=op, is_valid=False,
            rejection_reason=(
                f"Active edge exists — use update_edge_confidence on {existing['id']}."
            ),
            duplicate_edge_id=existing["id"],
        )

    return ValidationResult(op=op, is_valid=True)


def _validate_update_edge_confidence(op):
    """
    Validate an update_edge_confidence operation.

    Checks:
      1. edge_id exists in the graph
      2. The edge is currently active

    Returns ValidationResult with is_valid=True if both checks pass.
    """
    edge = kg.get_edge_by_id(op.edge_id)
    if not edge:
        return ValidationResult(
            op=op, is_valid=False,
            rejection_reason=f"edge_id={op.edge_id!r} not found.",
        )
    if not edge["active"]:
        return ValidationResult(
            op=op, is_valid=False,
            rejection_reason="Edge is inactive.",
        )
    return ValidationResult(op=op, is_valid=True)


def _validate_deactivate_edge(op):
    """
    Validate a deactivate_edge operation.

    Checks:
      1. edge_id exists in the graph
      2. The edge is currently active (can't deactivate an already inactive edge)

    Returns ValidationResult with is_valid=True if both checks pass.
    """
    edge = kg.get_edge_by_id(op.edge_id)
    if not edge:
        return ValidationResult(
            op=op, is_valid=False,
            rejection_reason=f"edge_id={op.edge_id!r} not found.",
        )
    if not edge["active"]:
        return ValidationResult(
            op=op, is_valid=False,
            rejection_reason="Edge is already inactive.",
        )
    return ValidationResult(op=op, is_valid=True)


def _validate_add_alias(op):
    """
    Validate an add_alias operation.

    Checks:
      1. node_id exists in the graph
      2. alias is non-empty after stripping whitespace

    Returns ValidationResult with is_valid=True if both checks pass.
    """
    node = kg.get_node_by_id(op.node_id)
    if not node:
        return ValidationResult(
            op=op, is_valid=False,
            rejection_reason=f"node_id={op.node_id!r} not found.",
        )
    if not op.alias or not op.alias.strip():
        return ValidationResult(
            op=op, is_valid=False,
            rejection_reason="alias is empty.",
        )
    return ValidationResult(op=op, is_valid=True)


# Public API

def validate_operations(proposed):
    """
    Validate all proposed operations and return their validation results.

    Each operation is dispatched to the appropriate validator based on
    its op_type. Unknown operation types are rejected immediately.
    Exceptions during validation are caught and logged — they produce
    a rejection rather than failing the entire batch.

    Args:
      proposed: list[ProposedOperation] from operation_proposer.

    Returns:
      list[ValidationResult] — one per proposed operation, in the same order.
      Empty list if proposed is empty or None.
    """
    if not proposed:
        return []

    fns = {
        "insert_edge":            _validate_insert_edge,
        "update_edge_confidence": _validate_update_edge_confidence,
        "deactivate_edge":        _validate_deactivate_edge,
        "add_alias":              _validate_add_alias,
    }

    results = []
    for op in proposed:
        try:
            fn = fns.get(op.op_type)
            if fn:
                results.append(fn(op))
            else:
                results.append(ValidationResult(
                    op=op, is_valid=False,
                    rejection_reason=f"Unknown op_type={op.op_type!r}.",
                ))
        except Exception:
            log.exception(
                "validate_operations: error on op_type=%s.", op.op_type,
            )
            results.append(ValidationResult(
                op=op, is_valid=False,
                rejection_reason="Exception during validation.",
            ))

    return results
