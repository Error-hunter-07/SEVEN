from __future__ import annotations
from dataclasses import dataclass, field
import Database.kg_db_client as kg
from GlobalHelpers.logger import get_logger
from KnowledgeGraph.constants import RELATION_TYPES, SYMMETRIC_RELATIONS, MIN_EDGE_CONFIDENCE, CONFIDENCE_MIN, CONFIDENCE_MAX
from KnowledgeGraph.operation_proposer import ProposedOperation
log = get_logger(__name__)

@dataclass
class ValidationResult:
    op: ProposedOperation
    is_valid: bool
    rejection_reason: str = ""
    duplicate_edge_id: str = ""

def _validate_insert_edge(op):
    src = kg.get_node_by_id(op.source_id)
    if not src: return ValidationResult(op=op, is_valid=False, rejection_reason=f"source_id={op.source_id!r} not found.")
    tgt = kg.get_node_by_id(op.target_id)
    if not tgt: return ValidationResult(op=op, is_valid=False, rejection_reason=f"target_id={op.target_id!r} not found.")
    if op.source_id == op.target_id: return ValidationResult(op=op, is_valid=False, rejection_reason="Self-loop rejected.")
    if op.relation not in RELATION_TYPES: return ValidationResult(op=op, is_valid=False, rejection_reason=f"relation={op.relation!r} not in RELATION_TYPES.")
    if op.confidence < MIN_EDGE_CONFIDENCE: return ValidationResult(op=op, is_valid=False, rejection_reason=f"confidence={op.confidence:.3f} below MIN={MIN_EDGE_CONFIDENCE}.")
    if op.relation in SYMMETRIC_RELATIONS and op.source_id > op.target_id:
        op.source_id, op.target_id = op.target_id, op.source_id
    existing = kg.get_edge_between(op.source_id, op.target_id, relation=op.relation, active_only=True)
    if existing: return ValidationResult(op=op, is_valid=False, rejection_reason=f"Active edge exists — use update_edge_confidence on {existing['id']}.", duplicate_edge_id=existing["id"])
    return ValidationResult(op=op, is_valid=True)

def _validate_update_edge_confidence(op):
    edge = kg.get_edge_by_id(op.edge_id)
    if not edge: return ValidationResult(op=op, is_valid=False, rejection_reason=f"edge_id={op.edge_id!r} not found.")
    if not edge["active"]: return ValidationResult(op=op, is_valid=False, rejection_reason="Edge is inactive.")
    return ValidationResult(op=op, is_valid=True)

def _validate_deactivate_edge(op):
    edge = kg.get_edge_by_id(op.edge_id)
    if not edge: return ValidationResult(op=op, is_valid=False, rejection_reason=f"edge_id={op.edge_id!r} not found.")
    if not edge["active"]: return ValidationResult(op=op, is_valid=False, rejection_reason="Edge is already inactive.")
    return ValidationResult(op=op, is_valid=True)

def _validate_add_alias(op):
    node = kg.get_node_by_id(op.node_id)
    if not node: return ValidationResult(op=op, is_valid=False, rejection_reason=f"node_id={op.node_id!r} not found.")
    if not op.alias or not op.alias.strip(): return ValidationResult(op=op, is_valid=False, rejection_reason="alias is empty.")
    return ValidationResult(op=op, is_valid=True)

def validate_operations(proposed):
    if not proposed: return []
    fns = {"insert_edge": _validate_insert_edge, "update_edge_confidence": _validate_update_edge_confidence,
           "deactivate_edge": _validate_deactivate_edge, "add_alias": _validate_add_alias}
    results = []
    for op in proposed:
        try:
            fn = fns.get(op.op_type)
            results.append(fn(op) if fn else ValidationResult(op=op, is_valid=False, rejection_reason=f"Unknown op_type={op.op_type!r}."))
        except Exception:
            log.exception("validate_operations: error on op_type=%s.", op.op_type)
            results.append(ValidationResult(op=op, is_valid=False, rejection_reason="Exception during validation."))
    return results