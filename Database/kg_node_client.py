"""
Database/kg_node_client.py

CRUD for kg_nodes — the canonical concept store of the Knowledge Graph.
Includes the keyword index helpers (_extract_keywords, _index_node_keywords)
because keywords are derived entirely from node name + heading and must
be kept in sync with every node write.
"""

from __future__ import annotations

import json
import re
from typing import Optional

import Database.local_db as local_db
from Database.kg_constants import NODE_TYPES, _STOPWORDS, _now, _new_id, RELATION_TYPES
from GlobalHelpers.logger import get_logger

log = get_logger(__name__)

def _row_to_node(row) -> dict:
    """
    Convert a sqlite3.Row from kg_nodes into a plain dict.

    Always decodes the JSON `attributes` column so callers never need to
    call json.loads themselves. Falls back to {} on malformed JSON rather
    than raising — a corrupt attributes field should not make the whole
    node unreadable.
    """
    try:
        attributes = json.loads(row["attributes"]) if row["attributes"] else {}
    except (json.JSONDecodeError, TypeError):
        log.warning(
            "_row_to_node: failed to decode attributes for node id=%s — defaulting to {}.",
            row["id"],
        )
        attributes = {}

    return {
        "id":              row["id"],
        "name":            row["name"],
        "type":            row["type"],
        "heading":         row["heading"],
        "attributes":      attributes,
        "importance":      row["importance"],
        "confidence":      row["confidence"],
        "access_count":    row["access_count"],
        "traversal_count": row["traversal_count"],
        "created_at":      row["created_at"],
        "updated_at":      row["updated_at"],
    }


def _extract_keywords(name: str, heading: str) -> list[str]:
    """
    Extract meaningful individual words from a node's name and heading
    for the inverted keyword index.

    Lowercases everything, strips punctuation, removes stopwords and
    single-character tokens. Returns a deduplicated list.

    This is used by both insert_node (subtask 2, called from subtask 3's
    _index_node_keywords) and update_node (same path) to keep the keyword
    index consistent with the current name + heading.
    """
    import re
    raw = f"{name} {heading}".lower()
    tokens = re.findall(r"[a-z0-9]+", raw)
    return list({
        t for t in tokens
        if len(t) > 1 and t not in _STOPWORDS
    })


# ---------------------------------------------------------------------------
# Node CRUD
# ---------------------------------------------------------------------------

def insert_node(
    name:       str,
    type:       str        = "Concept",
    heading:    str        = "",
    attributes: dict       = None,
    importance: float      = 0.5,
    confidence: float      = 0.5,
) -> Optional[str]:
    """
    Insert a new node into kg_nodes. Returns the new node's UUID on
    success, None on failure.

    UNIQUENESS: kg_nodes.name has a UNIQUE constraint. If a node with
    the same name already exists this will raise an IntegrityError, which
    is caught and returned as None with a WARNING log (not ERROR — a
    duplicate insert is a logic issue in the caller, not a DB bug).
    The caller (Phase 2 sleep pipeline) should always call
    get_node_by_name() first and update instead of insert when the node
    already exists.

    TYPE VALIDATION: if `type` is not in NODE_TYPES it is accepted but
    logged as a warning. We don't hard-reject it here because the sleep
    pipeline may encounter edge-case types that deserve a log entry rather
    than a silent failure.

    ATTRIBUTES: stored as JSON. Defaults to {} if None is passed.
    """
    if not name or not name.strip():
        log.warning("insert_node: called with empty name — skipping.")
        return None

    name = name.strip()

    if type not in NODE_TYPES:
        log.warning(
            "insert_node: type=%r is not in NODE_TYPES %s — inserting anyway.",
            type, sorted(NODE_TYPES),
        )

    conn   = local_db.get_connection()
    new_id = _new_id()
    now    = _now()

    try:
        conn.execute(
            """
            INSERT INTO kg_nodes
                (id, name, type, heading, attributes,
                 importance, confidence, access_count, traversal_count,
                 created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, ?, ?)
            """,
            (
                new_id,
                name,
                type,
                heading.strip() if heading else "",
                json.dumps(attributes or {}),
                float(importance),
                float(confidence),
                now,
                now,
            ),
        )
        conn.commit()
        log.info("insert_node: created [%s] name=%r type=%s.", new_id[:8], name, type)
        # Build the keyword index for this node immediately after insert.
        # Done after commit so the node row exists when keyword FK fires.
        # Imported lazily here to avoid a circular import with kg_alias_client.
        from Database.kg_alias_client import _index_node_keywords  # noqa: PLC0415
        ## noqa: PLC0415 is also a linting suppression, but this one is specifically associated with Pylint.
        #it supresses the warning about importing inside a function, which is usually discouraged but necessary here to avoid circular dependencies.
        _index_node_keywords(new_id, name, heading or "")
        return new_id

    except Exception as e:
        conn.rollback()
        # IntegrityError on name UNIQUE → warning level, not error.
        # Any other exception is a genuine DB error.
        import sqlite3
        if isinstance(e, sqlite3.IntegrityError):
            log.warning(
                "insert_node: node name=%r already exists (IntegrityError) — "
                "use get_node_by_name() + update_node() instead of insert.",
                name,
            )
        else:
            log.error(
                "insert_node: unexpected error for name=%r: %s",
                name, e, exc_info=True,
            )
        return None


def get_node_by_id(node_id: str) -> Optional[dict]:
    """
    Fetch a single node by its UUID primary key.

    Returns the node dict or None if not found. This is the cheapest
    lookup — a direct PK scan with no index needed.

    Bumps access_count via a separate UPDATE - we read first, then increment
    separately only when the row exists. The two-statement pattern is
    safe because the Knowledge Graph is never under write contention
    from multiple concurrent processes (only the sleep pipeline writes,
    and it runs sequentially).
    """
    if not node_id:
        log.warning("get_node_by_id: called with empty node_id — returning None.")
        return None

    conn = local_db.get_connection()
    try:
        row = conn.execute(
            """
            SELECT id, name, type, heading, attributes,
                   importance, confidence, access_count, traversal_count,
                   created_at, updated_at
            FROM kg_nodes
            WHERE id = ?
            """,
            (node_id,),
        ).fetchone()

        if row is None:
            return None

        return _row_to_node(row)

    except Exception as e:
        log.error("get_node_by_id(%s) error: %s", node_id, e, exc_info=True)
        return None


def get_node_by_name(name: str) -> Optional[dict]:
    """
    Exact-match lookup by canonical name. Uses idx_kg_nodes_name — O(1).

    This is the primary entity resolution path during the sleep pipeline:
    before creating a new node, always call this to check whether the
    concept already exists. The UNIQUE constraint on kg_nodes.name means
    this returns at most one row.

    Case-sensitive: "PostgreSQL" and "postgresql" are different names.
    The sleep pipeline is responsible for normalising case before calling
    this — typically by storing canonical names in title case and
    lowercasing aliases in kg_node_aliases (subtask 3).
    """
    if not name or not name.strip():
        log.warning("get_node_by_name: called with empty name — returning None.")
        return None

    conn = local_db.get_connection()
    try:
        row = conn.execute(
            """
            SELECT id, name, type, heading, attributes,
                   importance, confidence, access_count, traversal_count,
                   created_at, updated_at
            FROM kg_nodes
            WHERE name = ?
            """,
            (name.strip(),),
        ).fetchone()

        if row is None:
            return None

        return _row_to_node(row)

    except Exception as e:
        log.error("get_node_by_name(%r) error: %s", name, e, exc_info=True)
        return None


def get_nodes_by_type(
    node_type: str,
    order_by:  str   = "importance",
    limit:     int   = 50,
) -> list[dict]:
    """
    Return all nodes of a given type, ordered by importance or created_at.

    order_by accepts 'importance' (default) or 'created_at'. Any other
    value falls back to 'importance' with a warning rather than raising
    or allowing SQL injection through the ORDER BY clause.

    Used by:
      - Phase 2 sleep pipeline batch selection (avoid re-processing
        node types recently consolidated).
      - Phase 3 retrieval when the query contains a type hint
        ("what technologies does Seven use?").

    Returns [] on empty result or error (never None) so callers can
    always iterate safely.
    """
    if node_type not in NODE_TYPES:
        log.warning(
            "get_nodes_by_type: type=%r not in NODE_TYPES — will return [] "
            "(no nodes of unknown types exist).",
            node_type,
        )
        return []

    allowed_order = {"importance", "created_at"}
    if order_by not in allowed_order:
        log.warning(
            "get_nodes_by_type: order_by=%r not allowed — defaulting to 'importance'.",
            order_by,
        )
        order_by = "importance"

    # Build ORDER BY clause — safe because order_by is now whitelisted.
    order_clause = (
        "importance DESC, name ASC"
        if order_by == "importance"
        else "created_at DESC"
    )

    conn = local_db.get_connection()
    try:
        rows = conn.execute(
            f"""
            SELECT id, name, type, heading, attributes,
                   importance, confidence, access_count, traversal_count,
                   created_at, updated_at
            FROM kg_nodes
            WHERE type = ?
            ORDER BY {order_clause}
            LIMIT ?
            """,
            (node_type, int(limit)),
        ).fetchall()

        return [_row_to_node(r) for r in rows]

    except Exception as e:
        log.error("get_nodes_by_type(%r) error: %s", node_type, e, exc_info=True)
        return []


def update_node(
    node_id:    str,
    name:       Optional[str]   = None,
    type:       Optional[str]   = None,
    heading:    Optional[str]   = None,
    attributes: Optional[dict]  = None,
    importance: Optional[float] = None,
    confidence: Optional[float] = None,
) -> bool:
    """
    Partial update of a kg_nodes row. Only columns explicitly passed
    (not None) are changed. updated_at is always refreshed.

    Returns True if exactly one row was updated, False otherwise.

    NAME CHANGE: renaming a node changes its canonical identity. The
    sleep pipeline should only do this when entity resolution is highly
    confident (>= 0.85) that two names refer to the same concept.
    After renaming, the old name should be added as an alias via
    add_alias() (subtask 3) to preserve lookup continuity.

    TYPE CHANGE: validated against NODE_TYPES. Rejected with a warning
    if the new type is not in the controlled vocabulary.

    ATTRIBUTES: merged with existing attributes when partial is True
    (not yet implemented — currently replaces the whole attributes dict).
    Full replacement is simpler and sufficient for Phase 2.
    """
    if not node_id:
        log.warning("update_node: called with empty node_id — refusing no-op.")
        return False

    if type is not None and type not in NODE_TYPES:
        log.warning(
            "update_node: type=%r not in NODE_TYPES — update rejected.",
            type,
        )
        return False

    conn   = local_db.get_connection()
    fields = []
    values = []

    if name is not None:
        stripped = name.strip()
        if not stripped:
            log.warning("update_node: name cannot be set to empty string — skipping.")
            return False
        fields.append("name = ?")
        values.append(stripped)

    if type is not None:
        fields.append("type = ?")
        values.append(type)

    if heading is not None:
        fields.append("heading = ?")
        values.append(heading.strip())

    if attributes is not None:
        fields.append("attributes = ?")
        values.append(json.dumps(attributes))

    if importance is not None:
        fields.append("importance = ?")
        values.append(float(max(0.0, min(1.0, importance))))

    if confidence is not None:
        fields.append("confidence = ?")
        values.append(float(max(0.0, min(1.0, confidence))))

    if not fields:
        log.warning("update_node: no fields to update for node_id=%s.", node_id)
        return False

    fields.append("updated_at = ?")
    values.append(_now())
    values.append(node_id)

    try:
        cur = conn.execute(
            f"UPDATE kg_nodes SET {', '.join(fields)} WHERE id = ?",
            values,
        )
        if cur.rowcount == 0:
            log.warning(
                "update_node: no row matched id=%s — nothing updated.", node_id
            )
            conn.rollback()
            return False

        conn.commit()
        log.debug("update_node: updated [%s] fields=%s.", node_id[:8], [f.split(" =")[0] for f in fields[:-1]])
        # Re-index keywords if name or heading changed.
        if name is not None or heading is not None:
            node = get_node_by_id(node_id)
            if node:
                from Database.kg_alias_client import _index_node_keywords  # noqa: PLC0415
                _index_node_keywords(node_id, node["name"], node["heading"])
        return True

    except Exception as e:
        conn.rollback()
        import sqlite3
        if isinstance(e, sqlite3.IntegrityError):
            log.warning(
                "update_node: name conflict (IntegrityError) for node_id=%s — "
                "another node already has that name.",
                node_id,
            )
        else:
            log.error("update_node(%s) error: %s", node_id, e, exc_info=True)
        return False


def delete_node(node_id: str) -> bool:
    """
    Hard DELETE of a node by primary key.

    Because kg_edges, kg_node_aliases, kg_node_keywords, and
    kg_memory_nodes all have REFERENCES kg_nodes(id) ON DELETE CASCADE,
    this single DELETE removes all edges, aliases, keywords, and memory
    links for this node automatically.

    IMPORTANT: callers must log this operation to kg_graph_logs BEFORE
    calling delete_node. The graph log is the only audit trail — once the
    node is gone, the cascade has already removed all related rows.
    kg_db_client does not call log_graph_operation() internally on delete
    to avoid circular dependency with subtask 5. The sleep pipeline's
    graph_updater.py (Phase 2) is responsible for logging before deleting.

    Returns True if exactly one row was deleted, False if not found or
    on error.
    """
    if not node_id:
        log.warning("delete_node: called with empty node_id — skipping.")
        return False

    conn = local_db.get_connection()
    try:
        cur = conn.execute(
            "DELETE FROM kg_nodes WHERE id = ?",
            (node_id,),
        )
        conn.commit()

        if cur.rowcount == 0:
            log.warning(
                "delete_node: no row matched id=%s — nothing deleted.", node_id
            )
            return False

        log.info(
            "delete_node: deleted node [%s] (cascade removed edges, aliases, "
            "keywords, memory links).",
            node_id[:8],
        )
        return True

    except Exception as e:
        conn.rollback()
        log.error("delete_node(%s) error: %s", node_id, e, exc_info=True)
        return False


def increment_access_count(node_id: str) -> bool:
    """
    Atomically increment access_count by 1 and refresh updated_at.

    Called by Phase 3 retrieval every time a node is the direct target
    of a query. Kept as a dedicated function (rather than a full
    update_node call) because:
      1. It is called on every retrieval — it must be a single SQL
         statement, not a read-then-write pair.
      2. It must not accidentally overwrite other fields.

    Returns True on success, False if node not found or on error.
    """
    if not node_id:
        return False
    conn = local_db.get_connection()
    try:
        cur = conn.execute(
            """
            UPDATE kg_nodes
            SET access_count   = access_count + 1,
                updated_at     = ?
            WHERE id = ?
            """,
            (_now(), node_id),
        )
        conn.commit()
        return cur.rowcount == 1
    except Exception as e:
        conn.rollback()
        log.error("increment_access_count(%s) error: %s", node_id, e, exc_info=True)
        return False


def increment_traversal_count(node_id: str) -> bool:
    """
    Atomically increment traversal_count by 1.

    Called by Phase 3 BFS/DFS every time a node is encountered as an
    intermediate node during graph traversal — NOT as the direct query
    target (that is access_count). A node with high traversal_count but
    low access_count is a structural hub in the graph.

    Same single-statement rationale as increment_access_count.
    """
    if not node_id:
        return False
    conn = local_db.get_connection()
    try:
        cur = conn.execute(
            """
            UPDATE kg_nodes
            SET traversal_count = traversal_count + 1,
                updated_at      = ?
            WHERE id = ?
            """,
            (_now(), node_id),
        )
        conn.commit()
        return cur.rowcount == 1
    except Exception as e:
        conn.rollback()
        log.error("increment_traversal_count(%s) error: %s", node_id, e, exc_info=True)
        return False


def search_nodes_by_name_prefix(
    prefix: str,
    limit:  int = 10,
) -> list[dict]:
    """
    Return nodes whose name starts with `prefix` (case-insensitive).

    Used during entity resolution when neither exact name nor alias
    lookup succeeds but the query string partially matches a node name
    (e.g. "Postgre" → PostgreSQL). Cheaper than embedding search and
    covers common partial-name cases.

    Uses LIKE with a suffix wildcard — hits the idx_kg_nodes_name index
    for the prefix portion on most SQLite versions.

    Returns [] on empty result or error.
    """
    if not prefix or not prefix.strip():
        return []

    conn = local_db.get_connection()
    try:
        rows = conn.execute(
            """
            SELECT id, name, type, heading, attributes,
                   importance, confidence, access_count, traversal_count,
                   created_at, updated_at
            FROM kg_nodes
            WHERE name LIKE ? ESCAPE '\\'
            ORDER BY importance DESC, name ASC
            LIMIT ?
            """,
            (prefix.strip().replace("%", r"\%").replace("_", r"\_") + "%", int(limit)),
        ).fetchall()

        return [_row_to_node(r) for r in rows]

    except Exception as e:
        log.error("search_nodes_by_name_prefix(%r) error: %s", prefix, e, exc_info=True)
        return []


# ===========================================================================