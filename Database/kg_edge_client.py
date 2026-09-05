"""
Database/kg_edge_client.py

CRUD for kg_edges — directed relationships between Knowledge Graph nodes.
Edges use a controlled relation vocabulary (RELATION_TYPES) and support
an active/inactive flag for historical graph state preservation.
"""

from __future__ import annotations

import json
from typing import Optional

import Database.local_db as local_db
from Database.kg_constants import RELATION_TYPES, _now, _new_id
from GlobalHelpers.logger import get_logger

log = get_logger(__name__)

def _row_to_edge(row) -> dict:
    """
    Convert a sqlite3.Row from kg_edges into a plain dict.

    Decodes the JSON evidence_memory_ids column. Falls back to [] on
    malformed JSON — a corrupt evidence list should not make the edge
    unreadable.

    EDGE DICT SHAPE:
      {
        "id":                  str,
        "source_id":           str,
        "target_id":           str,
        "relation":            str,
        "confidence":          float,
        "weight":              float,
        "active":              bool,
        "evidence_memory_ids": list[str],
        "created_at":          str,
        "updated_at":          str,
      }
    """
    try:
        evidence = json.loads(row["evidence_memory_ids"]) if row["evidence_memory_ids"] else []
    except (json.JSONDecodeError, TypeError):
        log.warning(
            "_row_to_edge: failed to decode evidence_memory_ids for edge id=%s — defaulting to [].",
            row["id"],
        )
        evidence = []

    return {
        "id":                  row["id"],
        "source_id":           row["source_id"],
        "target_id":           row["target_id"],
        "relation":            row["relation"],
        "confidence":          row["confidence"],
        "weight":              row["weight"],
        "active":              bool(row["active"]),
        "evidence_memory_ids": evidence,
        "created_at":          row["created_at"],
        "updated_at":          row["updated_at"],
    }


def insert_edge(
    source_id:           str,
    target_id:           str,
    relation:            str,
    confidence:          float     = 0.5,
    weight:              float     = 0.5,
    evidence_memory_ids: list      = None,
) -> Optional[str]:
    """
    Insert a new directed edge: source --[relation]--> target.

    Returns the new edge UUID on success, None on failure.

    SELF-LOOPS: an edge where source_id == target_id is rejected.
    A concept cannot have a meaningful directed relationship with itself.

    RELATION VALIDATION: if relation is not in RELATION_TYPES it is
    rejected with a warning and None is returned. The sleep pipeline's
    validator (Phase 2) is responsible for mapping LLM-proposed free-text
    relations to the controlled vocabulary before calling insert_edge.

    DUPLICATE CHECK: the DB does not have a UNIQUE constraint on
    (source_id, target_id, relation) because the same pair can have
    multiple edges of the same type in different time periods
    (one active, one deactivated). The sleep pipeline should call
    get_edge_between() before insert to decide whether to create a new
    edge or update the existing active one.

    FK ENFORCEMENT: source_id and target_id must exist in kg_nodes.
    SQLite with PRAGMA foreign_keys=ON enforces this and raises
    IntegrityError if either node is missing.
    """
    if not source_id or not target_id or not relation:
        log.warning(
            "insert_edge: called with empty source_id, target_id, or relation — skipping."
        )
        return None

    if source_id == target_id:
        log.warning(
            "insert_edge: self-loop rejected (source_id == target_id == %s).",
            source_id,
        )
        return None

    if relation not in RELATION_TYPES:
        log.warning(
            "insert_edge: relation=%r not in RELATION_TYPES — rejected. "
            "Use KnowledgeGraph/constants.py to extend the vocabulary.",
            relation,
        )
        return None

    conn   = local_db.get_connection()
    new_id = _new_id()
    now    = _now()

    try:
        conn.execute(
            """
            INSERT INTO kg_edges
                (id, source_id, target_id, relation,
                 confidence, weight, active, evidence_memory_ids,
                 created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
            """,
            (
                new_id,
                source_id,
                target_id,
                relation,
                float(max(0.0, min(1.0, confidence))),
                float(max(0.0, min(1.0, weight))),
                json.dumps(evidence_memory_ids or []),
                now,
                now,
            ),
        )
        conn.commit()
        log.info(
            "insert_edge: [%s] %s --%s--> %s conf=%.2f.",
            new_id[:8], source_id[:8], relation, target_id[:8], confidence,
        )
        return new_id

    except Exception as e:
        conn.rollback()
        import sqlite3 as _sqlite3
        if isinstance(e, _sqlite3.IntegrityError):
            log.warning(
                "insert_edge: FK violation — source_id=%s or target_id=%s "
                "does not exist in kg_nodes.",
                source_id, target_id,
            )
        else:
            log.error(
                "insert_edge(%s --%s--> %s) error: %s",
                source_id, relation, target_id, e, exc_info=True,
            )
        return None


def get_edge_by_id(edge_id: str) -> Optional[dict]:
    """
    Fetch a single edge by its UUID primary key.
    Returns the edge dict or None if not found.
    """
    if not edge_id:
        return None
    conn = local_db.get_connection()
    try:
        row = conn.execute(
            """
            SELECT id, source_id, target_id, relation,
                   confidence, weight, active, evidence_memory_ids,
                   created_at, updated_at
            FROM kg_edges WHERE id = ?
            """,
            (edge_id,),
        ).fetchone()
        return _row_to_edge(row) if row else None
    except Exception as e:
        log.error("get_edge_by_id(%s) error: %s", edge_id, e, exc_info=True)
        return None


def get_edges_from(
    source_id:   str,
    active_only: bool = True,
    limit:       int  = 50,
) -> list[dict]:
    """
    Return all outgoing edges from a node, ordered by weight DESC.

    active_only=True (default) returns only current relationships.
    active_only=False includes deactivated historical edges — useful
    for answering "what did Seven think this node was connected to?".

    Uses idx_kg_edges_source — the primary graph traversal index.
    Returns [] on empty or error.
    """
    if not source_id:
        return []
    conn = local_db.get_connection()
    where_active = "AND active = 1" if active_only else ""
    try:
        rows = conn.execute(
            f"""
            SELECT id, source_id, target_id, relation,
                   confidence, weight, active, evidence_memory_ids,
                   created_at, updated_at
            FROM kg_edges
            WHERE source_id = ? {where_active}
            ORDER BY weight DESC, confidence DESC
            LIMIT ?
            """,
            (source_id, int(limit)),
        ).fetchall()
        return [_row_to_edge(r) for r in rows]
    except Exception as e:
        log.error("get_edges_from(%s) error: %s", source_id, e, exc_info=True)
        return []


def get_edges_to(
    target_id:   str,
    active_only: bool = True,
    limit:       int  = 50,
) -> list[dict]:
    """
    Return all incoming edges to a node, ordered by weight DESC.

    Reverse traversal — answers "what points at this node?".
    Uses idx_kg_edges_target.
    Returns [] on empty or error.
    """
    if not target_id:
        return []
    conn = local_db.get_connection()
    where_active = "AND active = 1" if active_only else ""
    try:
        rows = conn.execute(
            f"""
            SELECT id, source_id, target_id, relation,
                   confidence, weight, active, evidence_memory_ids,
                   created_at, updated_at
            FROM kg_edges
            WHERE target_id = ? {where_active}
            ORDER BY weight DESC, confidence DESC
            LIMIT ?
            """,
            (target_id, int(limit)),
        ).fetchall()
        return [_row_to_edge(r) for r in rows]
    except Exception as e:
        log.error("get_edges_to(%s) error: %s", target_id, e, exc_info=True)
        return []


def get_edge_between(
    source_id:   str,
    target_id:   str,
    relation:    Optional[str] = None,
    active_only: bool          = True,
) -> Optional[dict]:
    """
    Return the edge between two specific nodes, optionally filtered by
    relation type. Returns the highest-weight match when multiple edges
    exist between the same pair.

    Used by the sleep pipeline before insert_edge to determine whether
    to create a new edge or update the existing one's confidence/evidence.

    relation=None returns the strongest edge regardless of type.
    relation='uses' returns only a 'uses' edge between the two nodes.

    Uses idx_kg_edges_pair — O(1) for the (source, target) lookup.
    """
    if not source_id or not target_id:
        return None

    conn = local_db.get_connection()
    where_relation = "AND relation = ?" if relation else ""
    where_active   = "AND active = 1"   if active_only else ""
    params = [source_id, target_id]
    if relation:
        params.append(relation)

    try:
        row = conn.execute(
            f"""
            SELECT id, source_id, target_id, relation,
                   confidence, weight, active, evidence_memory_ids,
                   created_at, updated_at
            FROM kg_edges
            WHERE source_id = ? AND target_id = ?
                  {where_relation} {where_active}
            ORDER BY weight DESC, confidence DESC
            LIMIT 1
            """,
            params,
        ).fetchone()
        return _row_to_edge(row) if row else None
    except Exception as e:
        log.error(
            "get_edge_between(%s, %s) error: %s",
            source_id, target_id, e, exc_info=True,
        )
        return None


def update_edge_confidence(
    edge_id:    str,
    confidence: float,
    weight:     Optional[float] = None,
) -> bool:
    """
    Update an edge's confidence score and optionally its weight.

    The most common edge mutation — called by the sleep pipeline when a
    new memory provides additional evidence for an existing relationship,
    increasing confidence. Also called when contradictory evidence
    appears, decreasing confidence.

    Both confidence and weight are clamped to [0.0, 1.0].
    updated_at is always refreshed.

    Returns True on success, False if edge not found or on error.
    """
    if not edge_id:
        log.warning("update_edge_confidence: empty edge_id — skipping.")
        return False

    conn   = local_db.get_connection()
    fields = ["confidence = ?", "updated_at = ?"]
    values = [float(max(0.0, min(1.0, confidence))), _now()]

    if weight is not None:
        fields.insert(1, "weight = ?")
        values.insert(1, float(max(0.0, min(1.0, weight))))

    values.append(edge_id)

    try:
        cur = conn.execute(
            f"UPDATE kg_edges SET {', '.join(fields)} WHERE id = ?",
            values,
        )
        if cur.rowcount == 0:
            log.warning(
                "update_edge_confidence: no edge matched id=%s.", edge_id
            )
            conn.rollback()
            return False
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        log.error(
            "update_edge_confidence(%s) error: %s", edge_id, e, exc_info=True
        )
        return False


def add_evidence_to_edge(edge_id: str, memory_id: str) -> bool:
    """
    Append a memory_id to an edge's evidence_memory_ids list.

    Reads the current list, appends if not already present, writes back.
    The read-modify-write is safe here because the sleep pipeline is
    single-threaded and never writes edges concurrently.

    Returns True if the memory_id was added, False if it was already
    present (idempotent) or on error.
    """
    if not edge_id or not memory_id:
        log.warning("add_evidence_to_edge: empty edge_id or memory_id — skipping.")
        return False

    edge = get_edge_by_id(edge_id)
    if edge is None:
        log.warning("add_evidence_to_edge: edge %s not found.", edge_id)
        return False

    current = edge["evidence_memory_ids"]
    if memory_id in current:
        log.debug(
            "add_evidence_to_edge: memory_id=%s already in evidence for edge [%s].",
            memory_id, edge_id[:8],
        )
        return False

    updated = current + [memory_id]
    conn = local_db.get_connection()
    try:
        cur = conn.execute(
            "UPDATE kg_edges SET evidence_memory_ids = ?, updated_at = ? WHERE id = ?",
            (json.dumps(updated), _now(), edge_id),
        )
        conn.commit()
        return cur.rowcount == 1
    except Exception as e:
        conn.rollback()
        log.error(
            "add_evidence_to_edge(%s, %s) error: %s",
            edge_id, memory_id, e, exc_info=True,
        )
        return False


def deactivate_edge(edge_id: str) -> bool:
    """
    Mark an edge as inactive (active=0) without deleting it.

    Used when a relationship is superseded or contradicted — for example,
    when "Seven uses SQLite" is replaced by "Seven uses PostgreSQL", the
    old SQLite edge is deactivated rather than deleted. This preserves
    the historical graph state and keeps the evidence_memory_ids intact.

    The deactivated edge remains queryable via active_only=False in
    get_edges_from / get_edges_to.

    Returns True on success, False if not found or already inactive.
    """
    if not edge_id:
        log.warning("deactivate_edge: empty edge_id — skipping.")
        return False

    conn = local_db.get_connection()
    try:
        cur = conn.execute(
            "UPDATE kg_edges SET active = 0, updated_at = ? WHERE id = ? AND active = 1",
            (_now(), edge_id),
        )
        if cur.rowcount == 0:
            log.warning(
                "deactivate_edge: edge %s not found or already inactive.",
                edge_id,
            )
            conn.rollback()
            return False
        conn.commit()
        log.info("deactivate_edge: edge [%s] deactivated.", edge_id[:8])
        return True
    except Exception as e:
        conn.rollback()
        log.error("deactivate_edge(%s) error: %s", edge_id, e, exc_info=True)
        return False


def reactivate_edge(edge_id: str) -> bool:
    """
    Restore an inactive edge to active status.

    Used when a previously superseded relationship becomes valid again —
    for example, reverting a migration. Less common than deactivate_edge
    but necessary for correctness.

    Returns True on success, False if not found or already active.
    """
    if not edge_id:
        return False

    conn = local_db.get_connection()
    try:
        cur = conn.execute(
            "UPDATE kg_edges SET active = 1, updated_at = ? WHERE id = ? AND active = 0",
            (_now(), edge_id),
        )
        if cur.rowcount == 0:
            log.warning(
                "reactivate_edge: edge %s not found or already active.", edge_id
            )
            conn.rollback()
            return False
        conn.commit()
        log.info("reactivate_edge: edge [%s] reactivated.", edge_id[:8])
        return True
    except Exception as e:
        conn.rollback()
        log.error("reactivate_edge(%s) error: %s", edge_id, e, exc_info=True)
        return False


def delete_edge(edge_id: str) -> bool:
    """
    Hard DELETE of an edge by primary key.

    Use this only to correct mistakes (wrong source, wrong target, wrong
    relation entered by a buggy LLM batch). For relationship changes
    over time, use deactivate_edge instead to preserve history.

    Callers must log to kg_graph_logs BEFORE calling this — same
    contract as delete_node.

    Returns True if exactly one row was deleted, False otherwise.
    """
    if not edge_id:
        log.warning("delete_edge: empty edge_id — skipping.")
        return False

    conn = local_db.get_connection()
    try:
        cur = conn.execute(
            "DELETE FROM kg_edges WHERE id = ?",
            (edge_id,),
        )
        conn.commit()
        if cur.rowcount == 0:
            log.warning(
                "delete_edge: no edge matched id=%s — nothing deleted.", edge_id
            )
            return False
        log.info("delete_edge: deleted edge [%s].", edge_id[:8])
        return True
    except Exception as e:
        conn.rollback()
        log.error("delete_edge(%s) error: %s", edge_id, e, exc_info=True)
        return False


# ===========================================================================