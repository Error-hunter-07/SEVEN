"""
KnowledgeGraph/sleep_scheduler.py

Entry point for the Knowledge Graph sleep pipeline.

TRIGGERED BY:
  /sleep      — process up to MAX_BATCHES_PER_SLEEP session batches
  /sleep N    — process up to N session batches

PIPELINE PER SESSION BUNDLE:
  1. memory_selector.get_next_batch()     → list[SessionBundle]
  2. entity_extractor.extract_entities_from_bundle(bundle) → ExtractionResult | None
  3. entity_resolver.resolve_entities(extraction)          → list[ResolutionResult]
  4. subgraph_retriever.fetch_subgraph(resolved)           → Subgraph
  5. operation_proposer.propose_operations(...)            → list[ProposedOperation] | None
  6. validator.validate_operations(proposed)               → list[ValidationResult]
  7. _execute_valid_ops(results)                           → writes to DB
  8. _link_memories_to_nodes(resolved, bundle)             → kg_memory_nodes
  9. kg_sleep_queue_client.mark_processed(session_id)      → stamps queue row

FAILURE HANDLING:
  extraction returns None  → session skipped (stays pending, retried next /sleep)
  extraction returns []    → LLM found nothing; mark_processed() called anyway
                             (the session was examined, just empty)
  proposal returns None    → session skipped (stays pending)
  proposal returns []      → nothing to write; mark_processed() called
  All validator rejections → logged; valid ops executed, session still marked done
  Duplicate edge rejection → auto-converted to update_edge_confidence

SLEEP REPORT:
  run_sleep_cycle() returns a SleepReport dataclass summarising everything
  that happened. cli.py prints summary_lines() after each /sleep call.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from Database.kg_sleep_queue_client import mark_processed, count_pending
import Database.kg_db_client as kg
from KnowledgeGraph.constants import MAX_BATCHES_PER_SLEEP, RELATION_DEFAULT_WEIGHTS
from KnowledgeGraph.memory_selector import get_next_batch, get_queue_status, SessionBundle
from KnowledgeGraph.entity_extractor import extract_entities_from_bundle, ExtractionResult
from KnowledgeGraph.entity_resolver import resolve_entities, ResolutionResult
from KnowledgeGraph.subgraph_retriever import fetch_subgraph
from KnowledgeGraph.operation_proposer import propose_operations
from KnowledgeGraph.validator import validate_operations
from GlobalHelpers.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Sleep report
# ---------------------------------------------------------------------------

@dataclass
class SleepReport:
    """
    Summary of one run_sleep_cycle() call.

    Returned to cli.py which prints summary_lines() after each /sleep.
    """
    batches_attempted:  int   = 0
    batches_completed:  int   = 0
    batches_skipped:    int   = 0
    sessions_processed: int   = 0
    nodes_created:      int   = 0
    nodes_matched:      int   = 0
    edges_inserted:     int   = 0
    edges_updated:      int   = 0
    edges_deactivated:  int   = 0
    aliases_added:      int   = 0
    ops_rejected:       int   = 0
    ops_converted:      int   = 0   # duplicate → update_edge_confidence conversions
    memories_linked:    int   = 0
    elapsed_seconds:    float = 0.0

    def summary_lines(self) -> list[str]:
        return [
            f"Batches:   {self.batches_completed}/{self.batches_attempted} completed, {self.batches_skipped} skipped",
            f"Sessions:  {self.sessions_processed} processed",
            f"Nodes:     {self.nodes_created} created, {self.nodes_matched} matched",
            f"Edges:     {self.edges_inserted} inserted, {self.edges_updated} updated, {self.edges_deactivated} deactivated",
            f"Aliases:   {self.aliases_added} added",
            f"Rejected:  {self.ops_rejected} ops ({self.ops_converted} auto-converted duplicates)",
            f"Time:      {self.elapsed_seconds:.1f}s",
        ]


# ---------------------------------------------------------------------------
# Internal: execute valid operations
# ---------------------------------------------------------------------------

def _execute_valid_ops(
    validation_results: list,
    report: SleepReport,
) -> None:
    """
    Execute every operation that passed validation, update report counts.

    Duplicate-edge rejections (duplicate_edge_id set) are auto-converted
    to update_edge_confidence using the existing edge id and the original
    confidence value — the information is not lost.
    """
    from KnowledgeGraph.validator import ValidationResult
    from KnowledgeGraph.operation_proposer import ProposedOperation

    for vr in validation_results:
        if vr.is_valid:
            op = vr.op
            try:
                if op.op_type == "insert_edge":
                    weight = RELATION_DEFAULT_WEIGHTS.get(op.relation, 0.5)
                    eid = kg.insert_edge(
                        source_id           = op.source_id,
                        target_id           = op.target_id,
                        relation            = op.relation,
                        confidence          = op.confidence,
                        weight              = weight,
                        evidence_memory_ids = op.evidence_memory_ids,
                    )
                    if eid:
                        report.edges_inserted += 1
                        kg.log_graph_operation(
                            "insert_edge", "edge", eid,
                            details={"relation": op.relation, "confidence": op.confidence},
                            source="sleep_pipeline",
                        )

                elif op.op_type == "update_edge_confidence":
                    ok = kg.update_edge_confidence(op.edge_id, op.confidence)
                    if ok:
                        report.edges_updated += 1

                elif op.op_type == "deactivate_edge":
                    ok = kg.deactivate_edge(op.edge_id)
                    if ok:
                        report.edges_deactivated += 1
                        kg.log_graph_operation(
                            "deactivate_edge", "edge", op.edge_id,
                            details={"reasoning": op.reasoning},
                            source="sleep_pipeline",
                        )

                elif op.op_type == "add_alias":
                    ok = kg.add_alias(op.node_id, op.alias)
                    if ok:
                        report.aliases_added += 1

            except Exception:
                log.exception("_execute_valid_ops: error executing op_type=%s.", op.op_type)

        else:
            report.ops_rejected += 1

            # Auto-convert duplicate-edge rejection to update_edge_confidence
            if vr.duplicate_edge_id and vr.op.op_type == "insert_edge":
                try:
                    ok = kg.update_edge_confidence(vr.duplicate_edge_id, vr.op.confidence)
                    if ok:
                        report.edges_updated += 1
                        report.ops_converted += 1
                        log.info(
                            "_execute_valid_ops: auto-converted duplicate insert_edge → "
                            "update_edge_confidence on edge [%s].",
                            vr.duplicate_edge_id[:8],
                        )
                except Exception:
                    log.exception("_execute_valid_ops: auto-convert failed for duplicate edge.")
            else:
                log.info(
                    "_execute_valid_ops: op_type=%s rejected — %s",
                    vr.op.op_type, vr.rejection_reason[:80],
                )


# ---------------------------------------------------------------------------
# Internal: link memories to resolved nodes
# ---------------------------------------------------------------------------

def _link_memories_to_nodes(
    resolved: list[ResolutionResult],
    bundle:   SessionBundle,
    report:   SleepReport,
) -> None:
    """
    Write kg_memory_nodes links: for each resolved node, link it to all
    memory ids from this session (episodic + semantic).

    This is what lets Phase 3 answer "why do you think SEVEN uses PostgreSQL?"
    by tracing the edge back to the session memories that established it.
    """
    memory_ids: list[str] = []
    if bundle.episodic_memory_id:
        memory_ids.append(bundle.episodic_memory_id)
    memory_ids.extend(bundle.semantic_memory_ids)

    if not memory_ids:
        return

    valid_resolved = [r for r in resolved if r.node_id]
    for res in valid_resolved:
        for mem_id in memory_ids:
            try:
                mem_type = "episodic" if mem_id == bundle.episodic_memory_id else "semantic"
                ok = kg.link_memory_to_node(
                    node_id     = res.node_id,
                    memory_id   = mem_id,
                    memory_type = mem_type,
                    relevance   = res.entity.confidence,
                )
                if ok:
                    report.memories_linked += 1
            except Exception:
                log.exception(
                    "_link_memories_to_nodes: failed to link node [%s] ↔ memory %s.",
                    res.node_id[:8], mem_id,
                )


# ---------------------------------------------------------------------------
# Internal: process one SessionBundle through the full pipeline
# ---------------------------------------------------------------------------

def _process_bundle(bundle: SessionBundle, report: SleepReport) -> bool:
    """
    Run the full pipeline for one SessionBundle.

    Returns True if the session should be marked as processed
    (includes the "LLM found nothing" case — empty result is still done).
    Returns False if a LLM call failed and the session should be retried.
    """
    session_id = bundle.session_id

    # Step 1: extract entities
    extraction = extract_entities_from_bundle(bundle)
    if extraction is None:
        log.warning("_process_bundle: extraction failed for session=%s — skipping.", session_id)
        return False   # retry

    if not extraction.entities:
        log.info("_process_bundle: no entities found in session=%s — marking processed.", session_id)
        return True    # nothing to do, but don't re-examine

    # Step 2: resolve entities to graph nodes
    resolved = resolve_entities(extraction)

    valid_resolved = [r for r in resolved if r.node_id]
    for r in valid_resolved:
        if r.is_new:
            report.nodes_created += 1
            kg.log_graph_operation(
                "insert_node", "node", r.node_id,
                details={"name": r.node_name, "type": r.entity.type,
                         "session_id": session_id},
                source="sleep_pipeline",
            )
        else:
            report.nodes_matched += 1

    if not valid_resolved:
        log.info("_process_bundle: all entities skipped/failed for session=%s.", session_id)
        return True

    # Step 3: fetch local subgraph for context
    subgraph = fetch_subgraph(resolved)

    # Step 4: propose graph operations
    proposed = propose_operations(resolved, subgraph, [bundle], extraction.candidate_relations)
    if proposed is None:
        log.warning("_process_bundle: proposal failed for session=%s — skipping.", session_id)
        return False   # retry

    # Step 5: validate
    if proposed:
        validation_results = validate_operations(proposed)

        # Step 6: execute valid ops
        _execute_valid_ops(validation_results, report)

    # Step 7: link memories to resolved nodes
    _link_memories_to_nodes(resolved, bundle, report)

    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_sleep_cycle(
    max_batches:    int  = MAX_BATCHES_PER_SLEEP,
    batch_size:     int  = 1,
    print_progress: bool = True,
) -> SleepReport:
    """
    Run the Knowledge Graph sleep pipeline for up to `max_batches` batches.

    Each batch processes ONE session (batch_size=1 by default). Processing
    sessions one at a time gives the entity resolver the best chance of
    building on prior work: session 1 creates nodes, session 2 matches them.

    Args:
      max_batches:    Max sessions to process in one /sleep call.
                      Defaults to MAX_BATCHES_PER_SLEEP from constants.py.
      batch_size:     Sessions per memory_selector call. Keep at 1.
      print_progress: Print progress to stdout (for CLI use).

    Returns:
      SleepReport summarising what happened.
    """
    report    = SleepReport()
    t_start   = time.time()
    pending   = count_pending()

    if pending == 0:
        if print_progress:
            print("[Sleep] Nothing to process — knowledge graph is up to date.")
        report.elapsed_seconds = time.time() - t_start
        return report

    if print_progress:
        print(f"[Sleep] {pending} session(s) pending. Processing up to {max_batches}...")

    for batch_num in range(max_batches):
        bundles = get_next_batch(batch_size=batch_size)
        if not bundles:
            log.info("run_sleep_cycle: no more pending sessions after %d batches.", batch_num)
            break

        bundle = bundles[0]   # batch_size=1
        session_id = bundle.session_id
        report.batches_attempted += 1

        if print_progress:
            remaining = count_pending()
            print(
                f"[Sleep] Processing session {batch_num + 1}/{max_batches} "
                f"({remaining} remaining): {session_id[:12]}...",
                flush=True,
            )

        try:
            should_mark = _process_bundle(bundle, report)
        except Exception:
            log.exception(
                "run_sleep_cycle: unhandled error processing session=%s — skipping.",
                session_id,
            )
            should_mark = False

        if should_mark:
            mark_processed(session_id)
            report.batches_completed += 1
            report.sessions_processed += 1
        else:
            report.batches_skipped += 1

    report.elapsed_seconds = time.time() - t_start

    if print_progress:
        print("[Sleep] Complete.")
        for line in report.summary_lines():
            print(f"  {line}")

    return report