"""
Database/kg_link_client.py

Memory-node link table (kg_memory_nodes) and graph audit log
(kg_graph_logs) for the Knowledge Graph.

  Memory links — bridge between ChromaDB memory ids and graph nodes.
                 Bidirectional: node → memories, memory → nodes.
  Graph log    — append-only audit trail of every graph mutation.
                 Never deleted; the log IS the version history.
"""

from __future__ import annotations

import json
from typing import Optional

import Database.local_db as local_db
from Database.kg_constants import _now, _new_id
from Database.kg_node_client import _row_to_node, get_node_by_id
from GlobalHelpers.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Row converters
# ---------------------------------------------------------------------------

def _row_to_memory_link(row) -> dict:
    """
    Convert a sqlite3.Row from kg_memory_nodes into a plain dict.

    MEMORY LINK DICT SHAPE:
      {
        "id":          str,
        "node_id":     str,
        "memory_id":   str,   # ChromaDB document id
        "memory_type": str,   # 'episodic' | 'semantic'
        "relevance":   float, # 0.0-1.0
        "created_at":  str,
      }
    """
    return {
        "id":          row["id"],
        "node_id":     row["node_id"],
        "memory_id":   row["memory_id"],
        "memory_type": row["memory_type"],
        "relevance":   row["relevance"],
        "created_at":  row["created_at"],
    }


def _row_to_log_entry(row) -> dict:
    """
    Convert a sqlite3.Row from kg_graph_logs into a plain dict.

    LOG ENTRY DICT SHAPE:
      {
        "id":          str,
        "operation":   str,   # insert_node | insert_edge | update_node | …
        "entity_type": str,   # 'node' | 'edge' | 'alias' | 'keyword' | 'memory_link'
        "entity_id":   str,
        "details":     dict,  # decoded from JSON
        "source":      str,   # 'sleep_pipeline' | 'manual' | 'validator' | 'migration'
        "created_at":  str,
      }
    """
    try:
        details = json.loads(row["details"]) if row["details"] else {}
    except (json.JSONDecodeError, TypeError):
        details = {}

    return {
        "id":          row["id"],
        "operation":   row["operation"],
        "entity_type": row["entity_type"],
        "entity_id":   row["entity_id"],
        "details":     details,
        "source":      row["source"],
        "created_at":  row["created_at"],
    }


# ---------------------------------------------------------------------------
# Memory-node link functions
# ---------------------------------------------------------------------------

# Valid memory_type values — controls which ChromaDB collection Phase 3
# queries when fetching the raw memory behind a graph concept.
MEMORY_TYPES = frozenset({"episodic", "semantic"})

# Valid graph log operation names.
LOG_OPERATIONS = frozenset({
    "insert_node",
    "update_node",
    "delete_node",
    "insert_edge",
    "update_edge",
    "deactivate_edge",
    "reactivate_edge",
    "delete_edge",
    "add_alias",
    "delete_alias",
    "link_memory",
    "unlink_memory",
    "merge_node",
})


def link_memory_to_node(
    node_id:     str,
    memory_id:   str,
    memory_type: str   = "episodic",
    relevance:   float = 0.5,
) -> Optional[str]:
    """
    Create a link between a ChromaDB memory and a kg_nodes node.

    Returns the new link UUID on success, None on failure.

    DUPLICATE HANDLING: (node_id, memory_id) has a UNIQUE constraint.
    If the link already exists, returns None with a DEBUG log — not an
    error, since sleep batches can encounter the same memory multiple
    times across runs. Callers that need the existing link id should
    call get_memory_links_for_node() first.

    MEMORY_TYPE VALIDATION: rejects any value not in {'episodic', 'semantic'}.
    This controls which ChromaDB collection Phase 3 queries for evidence.

    RELEVANCE: clamped to [0.0, 1.0]. Represents how central this concept
    is to the memory — 1.0 means the memory is entirely about this node,
    0.2 means it is a passing mention.
    """
    if not node_id or not memory_id:
        log.warning("link_memory_to_node: empty node_id or memory_id — skipping.")
        return None

    if memory_type not in MEMORY_TYPES:
        log.warning(
            "link_memory_to_node: memory_type=%r not in %s — rejected.",
            memory_type, MEMORY_TYPES,
        )
        return None

    conn   = local_db.get_connection()
    new_id = _new_id()
    now    = _now()

    try:
        conn.execute(
            """
            INSERT INTO kg_memory_nodes
                (id, node_id, memory_id, memory_type, relevance, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                new_id,
                node_id,
                memory_id,
                memory_type,
                float(max(0.0, min(1.0, relevance))),
                now,
            ),
        )
        conn.commit()
        log.debug(
            "link_memory_to_node: node [%s] ← %s memory %s (relevance=%.2f).",
            node_id[:8], memory_type, memory_id, relevance,
        )
        return new_id

    except Exception as e:
        conn.rollback()
        import sqlite3 as _sqlite3
        if isinstance(e, _sqlite3.IntegrityError):
            # Either duplicate (node_id, memory_id) or bad node_id FK.
            # Distinguish by checking if the node exists.
            node_exists = get_node_by_id(node_id) is not None
            if node_exists:
                log.debug(
                    "link_memory_to_node: link already exists for node [%s] "
                    "← memory %s — skipping.",
                    node_id[:8], memory_id,
                )
            else:
                log.warning(
                    "link_memory_to_node: node_id=%s does not exist in kg_nodes "
                    "(FK violation).",
                    node_id,
                )
        else:
            log.error(
                "link_memory_to_node(node=%s, mem=%s) error: %s",
                node_id, memory_id, e, exc_info=True,
            )
        return None


def get_memory_links_for_node(
    node_id:     str,
    memory_type: Optional[str] = None,
    min_relevance: float       = 0.0,
    limit:       int           = 50,
) -> list[dict]:
    """
    Return all memory links for a node, ordered by relevance DESC.

    memory_type=None returns both episodic and semantic links.
    memory_type='episodic' or 'semantic' filters to that collection.

    min_relevance filters out low-relevance passing mentions when the
    caller only wants memories where this concept is central.

    Used by Phase 3 retrieval to answer "what memories support this node?"
    and by the sleep pipeline to skip memories already linked to a node.

    Returns [] on empty result or error.
    """
    if not node_id:
        return []

    if memory_type is not None and memory_type not in MEMORY_TYPES:
        log.warning(
            "get_memory_links_for_node: memory_type=%r not valid — returning [].",
            memory_type,
        )
        return []

    conn = local_db.get_connection()
    where_type = "AND memory_type = ?" if memory_type else ""
    params: list = [node_id, float(min_relevance)]
    if memory_type:
        params.insert(1, memory_type)

    # Rebuild params in correct WHERE order: node_id, [memory_type,] min_relevance, limit
    params = [node_id]
    if memory_type:
        params.append(memory_type)
    params.extend([float(min_relevance), int(limit)])

    try:
        rows = conn.execute(
            f"""
            SELECT id, node_id, memory_id, memory_type, relevance, created_at
            FROM kg_memory_nodes
            WHERE node_id = ? {where_type}
              AND relevance >= ?
            ORDER BY relevance DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [_row_to_memory_link(r) for r in rows]
    except Exception as e:
        log.error(
            "get_memory_links_for_node(%s) error: %s", node_id, e, exc_info=True
        )
        return []


def get_nodes_for_memory(
    memory_id:   str,
    memory_type: Optional[str] = None,
) -> list[dict]:
    """
    Return all nodes extracted from a given memory id, ordered by
    relevance DESC.

    The reverse direction of get_memory_links_for_node. Used by the
    sleep pipeline to check whether a memory has already been processed
    (if it returns non-empty, skip re-extraction).

    Returns full node dicts (joined with kg_nodes), not just link rows —
    so callers can immediately use the node data without a second query.

    Returns [] on empty result or error.
    """
    if not memory_id:
        return []

    conn = local_db.get_connection()
    where_type = "AND mn.memory_type = ?" if memory_type else ""
    params = [memory_id]
    if memory_type:
        params.append(memory_type)

    try:
        rows = conn.execute(
            f"""
            SELECT n.id, n.name, n.type, n.heading, n.attributes,
                   n.importance, n.confidence, n.access_count,
                   n.traversal_count, n.created_at, n.updated_at
            FROM kg_memory_nodes mn
            JOIN kg_nodes n ON n.id = mn.node_id
            WHERE mn.memory_id = ? {where_type}
            ORDER BY mn.relevance DESC
            """,
            params,
        ).fetchall()
        return [_row_to_node(r) for r in rows]
    except Exception as e:
        log.error(
            "get_nodes_for_memory(%s) error: %s", memory_id, e, exc_info=True
        )
        return []


def unlink_memory_from_node(node_id: str, memory_id: str) -> bool:
    """
    Remove a specific memory-node link.

    Used when a sleep batch determines that a memory was incorrectly
    linked to a node — for example, after entity resolution is corrected
    and the memory should point to a different node instead.

    Returns True if exactly one row was deleted, False otherwise.
    """
    if not node_id or not memory_id:
        log.warning("unlink_memory_from_node: empty node_id or memory_id — skipping.")
        return False

    conn = local_db.get_connection()
    try:
        cur = conn.execute(
            "DELETE FROM kg_memory_nodes WHERE node_id = ? AND memory_id = ?",
            (node_id, memory_id),
        )
        conn.commit()
        if cur.rowcount == 0:
            log.warning(
                "unlink_memory_from_node: no link found for node [%s] ← memory %s.",
                node_id[:8], memory_id,
            )
            return False
        log.debug(
            "unlink_memory_from_node: removed link node [%s] ← memory %s.",
            node_id[:8], memory_id,
        )
        return True
    except Exception as e:
        conn.rollback()
        log.error(
            "unlink_memory_from_node(%s, %s) error: %s",
            node_id, memory_id, e, exc_info=True,
        )
        return False


def get_unprocessed_memory_ids(
    all_memory_ids: list[str],
) -> list[str]:
    """
    Given a list of memory ids, return only those that have NO existing
    link in kg_memory_nodes — i.e. memories the sleep pipeline has not
    yet processed.

    This is the primary mechanism for avoiding re-extraction: before
    starting a sleep batch, the pipeline calls this to filter the
    candidate memory list down to genuinely new ones.

    Uses a single SQL query with an IN clause rather than N individual
    lookups. Safe against empty input (returns [] immediately).

    Returns [] if all memories are already linked, or on error.
    """
    if not all_memory_ids:
        return []

    conn = local_db.get_connection()
    placeholders = ",".join("?" * len(all_memory_ids))
    try:
        already_linked = {
            r["memory_id"]
            for r in conn.execute(
                f"SELECT DISTINCT memory_id FROM kg_memory_nodes WHERE memory_id IN ({placeholders})",
                all_memory_ids,
            ).fetchall()
        }
        return [mid for mid in all_memory_ids if mid not in already_linked]
    except Exception as e:
        log.error("get_unprocessed_memory_ids error: %s", e, exc_info=True)
        return []


# ---------------------------------------------------------------------------
# Graph audit log functions
# ---------------------------------------------------------------------------

def log_graph_operation(
    operation:   str,
    entity_type: str,
    entity_id:   str,
    details:     dict          = None,
    source:      str           = "sleep_pipeline",
) -> Optional[str]:
    """
    Append a record to the kg_graph_logs audit trail.

    MUST be called BEFORE the actual mutation is applied — this is the
    contract between log_graph_operation and the graph_updater (Phase 2).
    If the mutation fails after the log entry is written, the log still
    reflects the intent, which is useful for debugging.

    operation must be one of LOG_OPERATIONS. If not, the entry is still
    written but a WARNING is logged — unrecognised operations are not
    rejected because a strict enum would require a migration every time
    Phase 2 adds a new operation type.

    details should contain the full before/after state or the parameters
    used, encoded as a plain dict (will be JSON-serialised). Keep it
    human-readable: future debugging depends on these entries being
    interpretable without running code.

    Returns the new log entry UUID on success, None on failure.
    The log is append-only — there is no delete_log_entry() function.
    """
    if not operation or not entity_type or not entity_id:
        log.warning(
            "log_graph_operation: missing required field(s) — skipping."
        )
        return None

    if operation not in LOG_OPERATIONS:
        log.warning(
            "log_graph_operation: operation=%r not in LOG_OPERATIONS — "
            "logging anyway but consider adding it to the set.",
            operation,
        )

    conn   = local_db.get_connection()
    new_id = _new_id()
    now    = _now()

    try:
        conn.execute(
            """
            INSERT INTO kg_graph_logs
                (id, operation, entity_type, entity_id, details, source, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_id,
                operation,
                entity_type,
                entity_id,
                json.dumps(details or {}),
                source,
                now,
            ),
        )
        conn.commit()
        log.debug(
            "log_graph_operation: [%s] %s %s entity=[%s] source=%s.",
            new_id[:8], operation, entity_type, entity_id[:8], source,
        )
        return new_id

    except Exception as e:
        conn.rollback()
        log.error(
            "log_graph_operation(%s, %s) error: %s",
            operation, entity_id, e, exc_info=True,
        )
        return None


def get_log_entries_for_entity(
    entity_id: str,
    limit:     int = 50,
) -> list[dict]:
    """
    Return all log entries for a specific node or edge, newest first.

    Used to answer "what happened to this node?" — shows the full
    operation history in reverse chronological order.

    Uses idx_kg_logs_entity_id — O(log n).
    Returns [] on empty result or error.
    """
    if not entity_id:
        return []

    conn = local_db.get_connection()
    try:
        rows = conn.execute(
            """
            SELECT id, operation, entity_type, entity_id,
                   details, source, created_at
            FROM kg_graph_logs
            WHERE entity_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (entity_id, int(limit)),
        ).fetchall()
        return [_row_to_log_entry(r) for r in rows]
    except Exception as e:
        log.error(
            "get_log_entries_for_entity(%s) error: %s", entity_id, e, exc_info=True
        )
        return []


def get_recent_log_entries(
    limit:      int            = 100,
    operation:  Optional[str]  = None,
    source:     Optional[str]  = None,
) -> list[dict]:
    """
    Return the most recent log entries across all entities, newest first.

    Optional filters:
      operation='insert_node' — only entries for that operation type.
      source='sleep_pipeline' — only entries from that source.

    Used by the sleep pipeline to review what the last batch did, and
    by a future "undo last sleep batch" command that replays these
    entries in reverse.

    Returns [] on empty result or error.
    """
    conn = local_db.get_connection()

    conditions = []
    params: list = []

    if operation:
        conditions.append("operation = ?")
        params.append(operation)
    if source:
        conditions.append("source = ?")
        params.append(source)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    params.append(int(limit))

    try:
        rows = conn.execute(
            f"""
            SELECT id, operation, entity_type, entity_id,
                   details, source, created_at
            FROM kg_graph_logs
            {where}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [_row_to_log_entry(r) for r in rows]
    except Exception as e:
        log.error("get_recent_log_entries error: %s", e, exc_info=True)
        return []


def get_log_entry_count() -> int:
    """
    Return the total number of log entries. Used for monitoring and
    to decide when to archive old entries (future feature).
    Returns -1 on error.
    """
    conn = local_db.get_connection()
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM kg_graph_logs"
        ).fetchone()[0]
    except Exception as e:
        log.error("get_log_entry_count error: %s", e, exc_info=True)
        return -1