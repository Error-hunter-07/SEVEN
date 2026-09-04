from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Optional
import LLMEngine.llm_request_lock as llm_request_lock
from GlobalHelpers.config import settings
from GlobalHelpers.logger import get_logger
from KnowledgeGraph.constants import (VALID_OPERATION_TYPES, OPERATION_REQUIRED_FIELDS,
    CONFIDENCE_MIN, CONFIDENCE_MAX, OPERATION_PROPOSAL_TIMEOUT, OPERATION_PROPOSAL_MAX_TOKENS,
    PIPELINE_TEMPERATURE, build_operation_proposal_system, build_operation_proposal_user)
from KnowledgeGraph.entity_resolver import ResolutionResult
from KnowledgeGraph.memory_selector import SessionBundle
from KnowledgeGraph.subgraph_retriever import Subgraph
log = get_logger(__name__)
_SYSTEM_PROMPT = build_operation_proposal_system()

@dataclass
class ProposedOperation:
    op_type: str
    source_id: str = ""; target_id: str = ""; relation: str = ""
    evidence_memory_ids: list = field(default_factory=list)
    confidence: float = 0.0; edge_id: str = ""; reasoning: str = ""
    node_id: str = ""; alias: str = ""; raw: dict = field(default_factory=dict)

def _strip_fences(raw):
    if raw.startswith("```"):
        return "\n".join(l for l in raw.splitlines() if not l.strip().startswith("```")).strip()
    return raw

def _parse_operation(raw, valid_ids: set[str]):
    if not isinstance(raw, dict): return None
    op_type = str(raw.get("type") or "").strip()
    if not op_type or op_type not in VALID_OPERATION_TYPES: return None
    for f in OPERATION_REQUIRED_FIELDS.get(op_type, []):
        val = raw.get(f)
        if val is None or (isinstance(val, (str, list)) and not val): return None
    op = ProposedOperation(op_type=op_type, raw=raw)
    try:
        if op_type == "insert_edge":
            op.source_id = str(raw["source_id"]).strip(); op.target_id = str(raw["target_id"]).strip()
            if op.source_id not in valid_ids or op.target_id not in valid_ids:
                log.warning("propose_operations: hallucinated id — source=%r target=%r not in resolved set.",
                            op.source_id, op.target_id)
                return None
            op.relation = str(raw["relation"]).strip()
            op.confidence = float(max(CONFIDENCE_MIN, min(CONFIDENCE_MAX, raw.get("confidence", 0.5))))
            ev = raw.get("evidence_memory_ids") or []
            op.evidence_memory_ids = [str(m) for m in ev if m] if isinstance(ev, list) else []
        elif op_type == "update_edge_confidence":
            op.edge_id = str(raw["edge_id"]).strip()
            op.confidence = float(max(CONFIDENCE_MIN, min(CONFIDENCE_MAX, raw.get("confidence", 0.5))))
        elif op_type == "deactivate_edge":
            op.edge_id = str(raw["edge_id"]).strip(); op.reasoning = str(raw.get("reasoning") or "").strip()
        elif op_type == "add_alias":
            op.node_id = str(raw["node_id"]).strip(); op.alias = str(raw["alias"]).strip().lower()
    except Exception: return None
    return op

def _parse_response(raw, bundle, valid_ids: set[str]):
    raw = _strip_fences(raw)
    try: parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        #Fixed the silent failure here to log the error and raw response for debugging
        log.warning("propose_operations: JSON decode error: %s. Raw: %r", e, raw)
        return None

    if not isinstance(parsed, dict): return None
    raw_ops = parsed.get("operations")
    if raw_ops is None or not isinstance(raw_ops, list): return None
    if not raw_ops: return []
    return [op for op in (_parse_operation(r, valid_ids) for r in raw_ops) if op is not None]

def propose_operations(resolved, subgraph, bundles, candidate_relations=None):
    valid = [r for r in resolved if r.node_id]
    if not valid: return []
    # 1. CREATING THE SET OF VALID IDs HERE
    valid_ids = {r.node_id for r in valid}

    name_to_id = {r.node_name.lower(): r.node_id for r in valid}
    hints = []
    for cr in (candidate_relations or []):
        sid = name_to_id.get(cr.source.lower())
        tid = name_to_id.get(cr.target.lower())
        if sid and tid:
            hints.append(f"  {cr.source} --{cr.relation}--> {cr.target}  (conf={cr.confidence:.2f}: {cr.reasoning})")

    resolved_for_prompt = [{"name":r.node_name,"node_id":r.node_id,"type":r.entity.type,"heading":r.entity.heading,"is_new":r.is_new} for r in valid]
    # bundles is list[SessionBundle] — use first bundle for memory context
    bundle = bundles[0] if bundles else None
    mem_ids_texts = []
    if bundle:
        if bundle.episodic_memory_id:
            mem_ids_texts.append((bundle.episodic_memory_id, bundle.episode_text or bundle.conversation_text[:200]))
        for mid, text in zip(bundle.semantic_memory_ids, bundle.semantic_texts):
            mem_ids_texts.append((mid, text[:200]))
    user_content = build_operation_proposal_user(resolved_for_prompt, subgraph.text, mem_ids_texts, hints)
    try:
        response = llm_request_lock.post_completion(
            {
                "model": settings.llm_model, 
                "messages": [{
                        "role":"system",
                        "content":_SYSTEM_PROMPT
                    },
                    {
                        "role":"user",
                        "content":user_content
                    }],
             "temperature": PIPELINE_TEMPERATURE, 
             "max_tokens": OPERATION_PROPOSAL_MAX_TOKENS, 
             "chat_template_kwargs": {
                 "enable_thinking": False
                 }},
            role="main", 
            timeout=OPERATION_PROPOSAL_TIMEOUT
            )
        response.raise_for_status()
        raw = response.json().get("choices",[{}])[0].get("message",{}).get("content","").strip()
    except Exception as e:
        log.error("propose_operations: LLM call failed: %s", e, exc_info=True); return None
    if not raw: return None
    return _parse_response(raw, bundle, valid_ids)