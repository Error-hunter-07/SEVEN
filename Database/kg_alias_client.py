"""
Database/kg_alias_client.py

Alias and keyword index functions for the Knowledge Graph.
  Alias index  — exact alternative names, O(1) lookup.
  Keyword index — inverted index over node name + heading words.
"""

from __future__ import annotations

from typing import Optional

import Database.local_db as local_db
from Database.kg_constants import _STOPWORDS, _now, _new_id
from Database.kg_node_client import _extract_keywords, get_node_by_id, _row_to_node
from GlobalHelpers.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Internal: keyword index maintenance
# ---------------------------------------------------------------------------

def _index_node_keywords(node_id: str, name: str, heading: str) -> int:
    """
    (Re)build the keyword index rows for a single node.

    Called internally by insert_node and update_node — never called
    directly by the sleep pipeline. Deletes all existing keywords for
    the node first (via delete_keywords_for_node) then re-inserts the
    freshly extracted set. This makes update_node safe: changing a
    node's name or heading automatically keeps the keyword index current.

    Returns the count of keyword rows successfully inserted.
    """
    delete_keywords_for_node(node_id)
    keywords = _extract_keywords(name, heading)
    inserted = 0
    for kw in keywords:
        if add_keyword(node_id, kw):
            inserted += 1
    log.debug(
        "_index_node_keywords: [%s] indexed %d keyword(s): %s",
        node_id[:8], inserted, keywords,
    )
    return inserted


# ---------------------------------------------------------------------------
# Alias functions
# ---------------------------------------------------------------------------

def add_alias(node_id: str, alias: str) -> bool:
    """
    Add an alternative name for a node to the alias index.

    Aliases are stored lowercase regardless of input case so that all
    lookups via get_node_by_alias() are effectively case-insensitive
    without requiring SQLite COLLATE NOCASE (which can behave
    inconsistently with non-ASCII characters).

    Returns True on success, False if the alias already exists for this
    node (UNIQUE constraint) or on any other error. A duplicate alias is
    logged at DEBUG level — it is a common no-op during sleep batches,
    not an error.
    """
    if not node_id or not alias or not alias.strip():
        log.warning(
            "add_alias: called with empty node_id or alias — skipping."
        )
        return False

    normalised = alias.strip().lower()
    conn   = local_db.get_connection()
    new_id = _new_id()
    now    = _now()

    try:
        conn.execute(
            """
            INSERT INTO kg_node_aliases (id, node_id, alias, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (new_id, node_id, normalised, now),
        )
        conn.commit()
        log.debug(
            "add_alias: [%s] alias=%r added to node [%s].",
            new_id[:8], normalised, node_id[:8],
        )
        return True

    except Exception as e:
        conn.rollback()
        import sqlite3 as _sqlite3
        if isinstance(e, _sqlite3.IntegrityError):
            log.debug(
                "add_alias: alias=%r already exists for node [%s] — skipping.",
                normalised, node_id[:8],
            )
        else:
            log.error(
                "add_alias(node=%s, alias=%r) error: %s",
                node_id, alias, e, exc_info=True,
            )
        return False


def get_aliases_for_node(node_id: str) -> list[str]:
    """
    Return all aliases for a given node, sorted alphabetically.

    Used by Phase 3 retrieval to show the full name set for a node,
    and by the sleep pipeline to check whether a new alias already
    exists before calling add_alias.

    Returns [] on empty result or error.
    """
    if not node_id:
        return []
    conn = local_db.get_connection()
    try:
        rows = conn.execute(
            """
            SELECT alias FROM kg_node_aliases
            WHERE node_id = ?
            ORDER BY alias ASC
            """,
            (node_id,),
        ).fetchall()
        return [r["alias"] for r in rows]
    except Exception as e:
        log.error(
            "get_aliases_for_node(%s) error: %s", node_id, e, exc_info=True
        )
        return []


def get_node_by_alias(alias: str) -> Optional[dict]:
    """
    Exact alias lookup — returns the full node dict for the node that
    owns this alias, or None if no match.

    Lowercases the input before querying so callers don't need to
    normalise case themselves.

    When multiple nodes share the same alias (ambiguity), returns the
    one with the highest importance score. True ambiguity resolution
    (asking the LLM to choose) is the sleep pipeline's job — this
    function just gives the best single guess when a quick answer is
    needed.

    Uses idx_kg_aliases_alias — O(1).
    """
    if not alias or not alias.strip():
        log.warning("get_node_by_alias: called with empty alias — returning None.")
        return None

    normalised = alias.strip().lower()
    conn = local_db.get_connection()
    try:
        row = conn.execute(
            """
            SELECT n.id, n.name, n.type, n.heading, n.attributes,
                   n.importance, n.confidence, n.access_count,
                   n.traversal_count, n.created_at, n.updated_at
            FROM kg_node_aliases a
            JOIN kg_nodes n ON n.id = a.node_id
            WHERE a.alias = ?
            ORDER BY n.importance DESC
            LIMIT 1
            """,
            (normalised,),
        ).fetchone()

        if row is None:
            return None

        return _row_to_node(row)

    except Exception as e:
        log.error(
            "get_node_by_alias(%r) error: %s", alias, e, exc_info=True
        )
        return None


def delete_alias(node_id: str, alias: str) -> bool:
    """
    Remove a specific alias from a node.

    Used when entity resolution determines that an alias was incorrectly
    assigned, or when a node is renamed and the old name should no longer
    resolve to it.

    Does NOT remove the canonical name from kg_nodes.name — that is
    done via update_node. This only removes an entry from kg_node_aliases.

    Returns True if exactly one row was deleted, False otherwise.
    """
    if not node_id or not alias or not alias.strip():
        log.warning("delete_alias: called with empty node_id or alias — skipping.")
        return False

    normalised = alias.strip().lower()
    conn = local_db.get_connection()
    try:
        cur = conn.execute(
            "DELETE FROM kg_node_aliases WHERE node_id = ? AND alias = ?",
            (node_id, normalised),
        )
        conn.commit()
        if cur.rowcount == 0:
            log.debug(
                "delete_alias: alias=%r not found for node [%s].",
                normalised, node_id[:8],
            )
            return False
        log.debug(
            "delete_alias: alias=%r removed from node [%s].",
            normalised, node_id[:8],
        )
        return True
    except Exception as e:
        conn.rollback()
        log.error(
            "delete_alias(%s, %r) error: %s", node_id, alias, e, exc_info=True
        )
        return False


# ---------------------------------------------------------------------------
# Keyword index functions
# ---------------------------------------------------------------------------

def add_keyword(node_id: str, keyword: str) -> bool:
    """
    Add a single keyword index entry for a node.

    Typically called only by _index_node_keywords (which is called
    internally by insert_node and update_node). Direct callers should
    have a good reason — adding arbitrary keywords to a node bypasses
    the stopword filter and deduplication that _extract_keywords provides.

    Keyword is stored lowercase. Duplicate (node_id, keyword) pairs are
    silently ignored (UNIQUE constraint) — returns False for duplicates
    but does not log a warning since this is expected during rebuilds.
    """
    if not node_id or not keyword or not keyword.strip():
        return False

    normalised = keyword.strip().lower()
    if len(normalised) <= 1 or normalised in _STOPWORDS:
        return False

    conn   = local_db.get_connection()
    new_id = _new_id()
    now    = _now()

    try:
        conn.execute(
            """
            INSERT INTO kg_node_keywords (id, node_id, keyword, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (new_id, node_id, normalised, now),
        )
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        import sqlite3 as _sqlite3
        if not isinstance(e, _sqlite3.IntegrityError):
            log.error(
                "add_keyword(node=%s, kw=%r) error: %s",
                node_id, keyword, e, exc_info=True,
            )
        return False


def get_nodes_by_keyword(
    keyword: str,
    limit:   int = 20,
) -> list[dict]:
    """
    Return all nodes that have `keyword` in their keyword index,
    ordered by node importance DESC.

    The keyword is normalised (lowercased, stripped) before querying.
    Used as the third-priority candidate generation step after name and
    alias lookup both miss.

    For multi-word queries the caller should split the query into
    individual words and call this function once per word, then take
    the union or intersection of the result sets — this function only
    handles single-keyword lookups to keep each call O(k).

    Returns [] on empty result or error.
    """
    if not keyword or not keyword.strip():
        return []

    normalised = keyword.strip().lower()
    if normalised in _STOPWORDS or len(normalised) <= 1:
        return []

    conn = local_db.get_connection()
    try:
        rows = conn.execute(
            """
            SELECT n.id, n.name, n.type, n.heading, n.attributes,
                   n.importance, n.confidence, n.access_count,
                   n.traversal_count, n.created_at, n.updated_at
            FROM kg_node_keywords k
            JOIN kg_nodes n ON n.id = k.node_id
            WHERE k.keyword = ?
            ORDER BY n.importance DESC
            LIMIT ?
            """,
            (normalised, int(limit)),
        ).fetchall()
        return [_row_to_node(r) for r in rows]
    except Exception as e:
        log.error(
            "get_nodes_by_keyword(%r) error: %s", keyword, e, exc_info=True
        )
        return []


def get_keywords_for_node(node_id: str) -> list[str]:
    """
    Return all keywords indexed for a given node, sorted alphabetically.

    Mainly used for debugging, auditing, and verifying that insert_node
    and update_node correctly populated the keyword index.
    """
    if not node_id:
        return []
    conn = local_db.get_connection()
    try:
        rows = conn.execute(
            """
            SELECT keyword FROM kg_node_keywords
            WHERE node_id = ?
            ORDER BY keyword ASC
            """,
            (node_id,),
        ).fetchall()
        return [r["keyword"] for r in rows]
    except Exception as e:
        log.error(
            "get_keywords_for_node(%s) error: %s", node_id, e, exc_info=True
        )
        return []


def delete_keywords_for_node(node_id: str) -> int:
    """
    Delete ALL keyword index entries for a node.

    Called internally by _index_node_keywords before re-indexing, so
    that update_node (which changes name or heading) always results in a
    clean, current keyword set — no stale words from the old name linger.

    Returns the number of rows deleted (0 is valid — a new node has
    none yet). Returns -1 on error.
    """
    if not node_id:
        return 0
    conn = local_db.get_connection()
    try:
        cur = conn.execute(
            "DELETE FROM kg_node_keywords WHERE node_id = ?",
            (node_id,),
        )
        conn.commit()
        return cur.rowcount
    except Exception as e:
        conn.rollback()
        log.error(
            "delete_keywords_for_node(%s) error: %s", node_id, e, exc_info=True
        )
        return -1


# ===========================================================================