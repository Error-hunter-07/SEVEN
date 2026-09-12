"""
Database/local_db.py

Single embedded SQLite database for the app's local structured storage.
Holds two tables now: working_memory and active_sessions.

CHANGED: episodic_memory is no longer a SQLite table. It moved to its
own ChromaDB collection (MemoryManagement/episodic_memory/episodic_memory_store.py)
because its primary access pattern is semantic search ("what did we
discuss about X"), not exact-match lookup — the same reasoning that
keeps working_memory and active_sessions in SQLite (always looked up by
session_id/id, never by meaning) but puts semantic_memory in Chroma.
Running a parallel SQLite copy of the same data alongside the Chroma
collection would have been pure redundancy, not defense in depth.

CHANGED: active_sessions gained three columns to make crash recovery
genuinely useful instead of a last-resort guess:
  - chunk_summaries: rolling 5-turn summaries, written live during the
    session (see LLMEngine/chunk_summary_worker.py), so a crash mid-way
    through a long session still has real narrative content to recover
    from, not just whatever's in working_memory.
  - full_conversation: the raw message history, overwritten each turn,
    as the last-resort fallback if even chunk summarization hasn't
    caught up yet (e.g. a crash in the first 4 turns).
  - related_semantic_memory_ids: which semantic-memory facts were
    created during this session, persisted turn-by-turn instead of only
    living in an in-memory dict (SessionManager/session_memory_tracker.py)
    that a crash would simply lose.

ADDED (Knowledge Graph — Phase 1):
  Six new tables prefixed `kg_` live in the same SQLite file. The KG is
  a relationship layer over the existing memory stack — it never
  duplicates memory content, only links concepts to each other and back
  to the memories that support them.

  kg_nodes          — one row per stable concept (Person, Technology,
                       Project, …). The canonical entity store.
  kg_edges          — directed relationships between nodes.
                       source --[relation]--> target, with confidence,
                       weight, active flag, and evidence memory ids.
  kg_node_aliases   — alternative names for a node (postgres, pgsql,
                       "postgres db" all map to the PostgreSQL node).
                       Powers the O(1) alias index used during entity
                       resolution so the LLM never needs to scan nodes.
  kg_node_keywords  — inverted keyword index. Each meaningful word from
                       a node's name/heading maps to the node id. Lets
                       candidate generation find nodes without embeddings
                       for 95%+ of queries.
  kg_memory_nodes   — link table between episodic/semantic memory ids
                       and graph nodes. Lets Seven trace any graph
                       relationship back to the raw memories that
                       established it ("why do you think I use Postgres?").
  kg_graph_logs     — append-only audit log of every graph operation
                       (insert_node, insert_edge, merge_node, etc.).
                       Versioning without a full diff engine — the log
                       IS the version history.

  All six tables follow the same _ensure_*_schema() pattern as the
  existing tables and are registered in _ensure_schema() so they are
  created on startup alongside working_memory and active_sessions.

ADDED (Knowledge Graph — sleep queue):
  kg_sleep_queue     — durable hand-off table between a session ending
                       and the sleep pipeline processing it. Populated
                       in SessionManager/session_lifecycle.py's
                       on_session_end (and crash recovery's
                       _finalize_crashed_session) BEFORE the
                       active_sessions row is deleted, so the session's
                       episodic id, semantic memory ids, and narrative
                       text all survive past close_session(). The sleep
                       pipeline reads pending rows from here instead of
                       reconstructing session context from ChromaDB
                       metadata after the fact. A row is only deleted
                       (or its processed_at stamped) once the sleep
                       pipeline has actually consumed it, so
                       count_pending() also doubles as "how many
                       sessions are waiting to be processed".

SCHEMA INIT: each table has its own independent _ensure_*_schema()
function rather than one shared executescript() blob, so a DDL problem
in one table's block can't take a working, unrelated table's init down
with it.
"""

import sqlite3
import os
import threading
from GlobalHelpers.logger import get_logger

log = get_logger(__name__)

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "seven_local.db")

_local = threading.local()
_init_lock = threading.Lock()
_schema_ready = False


def _create_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.row_factory = sqlite3.Row
    return conn


def get_connection() -> sqlite3.Connection:
    if getattr(_local, "conn", None) is None:
        _local.conn = _create_connection()
    return _local.conn


def _ensure_working_memory_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS working_memory (
            id            TEXT PRIMARY KEY,
            session_id    TEXT NOT NULL,
            memory_type   TEXT,
            key           TEXT,
            value         TEXT,              -- JSON-encoded
            priority      REAL DEFAULT 0.5,
            relevance     REAL DEFAULT 0.5,
            created_at    TEXT NOT NULL,
            updated_at    TEXT NOT NULL,
            expires_at    TEXT,
            source        TEXT,
            tags          TEXT,               -- JSON-encoded list
            access_count  INTEGER DEFAULT 0,
            last_accessed TEXT,
            active        INTEGER DEFAULT 1
        );

        CREATE INDEX IF NOT EXISTS idx_working_memory_session
            ON working_memory (session_id, active);
        """
    )


def _ensure_active_sessions_schema(conn: sqlite3.Connection) -> None:
    """
    Crash-durability marker + live scratch table for the current session.
    Written at session start, updated on every turn, cleared on a clean
    session end. Any row still status='in_progress' at the NEXT
    process's startup belongs to a session that never got a clean
    shutdown — see SessionManager/session_lifecycle.py's crash-recovery
    sweep, which now has real material (chunk_summaries,
    full_conversation, related_semantic_memory_ids) to recover from
    instead of just a working_memory snippet.
    """
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS active_sessions (
            session_id                    TEXT PRIMARY KEY,
            started_at                    TEXT NOT NULL,
            last_turn_at                  TEXT,
            turn_count                    INTEGER DEFAULT 0,
            status                        TEXT NOT NULL DEFAULT 'in_progress',
            chunk_summaries                TEXT,   -- JSON list of strings, appended every 5 turns
            full_conversation             TEXT,   -- JSON list of {role, content} messages, overwritten each turn
            related_semantic_memory_ids   TEXT     -- JSON list of ChromaDB ids created this session
        );
        """
    )


def _ensure_kg_nodes_schema(conn: sqlite3.Connection) -> None:
    """
    kg_nodes — one row per stable concept in the knowledge graph.

    COLUMNS:
      id             — UUID primary key, permanent even if name changes.
      name           — canonical name of the concept (e.g. "PostgreSQL").
                       UNIQUE enforced so duplicate node creation is
                       caught at the DB level, not just in application code.
      type           — controlled vocabulary: Person | Project | Technology |
                       Organization | Place | Event | Concept.
                       Kept controlled to make entity resolution easier for
                       a local LLM — open-ended types produce inconsistent graphs.
      heading        — 4-5 word human-readable description ("Relational Database System").
                       Used during graph visualization and LLM subgraph prompts.
      attributes     — JSON blob for type-specific extra fields (e.g. language,
                       version, url). Kept flexible so the schema doesn't need
                       migration every time a new node type gains a new property.
      importance     — 0.0-1.0. Influenced by connected memory count, edge count,
                       retrieval frequency. Recalculated lazily per access during
                       Phase 3, not eagerly across all nodes (too expensive).
      confidence     — 0.0-1.0. How strongly Seven believes this node correctly
                       represents the underlying concept. Increases when multiple
                       independent memories support it; decreases on contradiction.
      access_count   — direct retrieval hits. Distinct from traversal_count.
      traversal_count— times encountered as an intermediate node during BFS/DFS.
                       A node with low access_count but high traversal_count is
                       a "hub" — important for graph topology, rarely queried directly.
      created_at     — ISO timestamp.
      updated_at     — ISO timestamp, refreshed on every write.

    INDEXES:
      idx_kg_nodes_name   — exact-match name lookup, O(1). Primary lookup path.
      idx_kg_nodes_type   — filter all nodes of a given type. Used during Sleep
                            Mode batch selection to avoid re-processing node types
                            that were recently consolidated.
      idx_kg_nodes_importance — ORDER BY importance DESC for ranked retrieval.
    """
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS kg_nodes (
            id               TEXT PRIMARY KEY,
            name             TEXT NOT NULL UNIQUE,
            type             TEXT NOT NULL DEFAULT 'Concept',
            heading          TEXT NOT NULL DEFAULT '',
            attributes       TEXT NOT NULL DEFAULT '{}',   -- JSON
            importance       REAL NOT NULL DEFAULT 0.5,
            confidence       REAL NOT NULL DEFAULT 0.5,
            access_count     INTEGER NOT NULL DEFAULT 0,
            traversal_count  INTEGER NOT NULL DEFAULT 0,
            created_at       TEXT NOT NULL,
            updated_at       TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_kg_nodes_name
            ON kg_nodes (name);

        CREATE INDEX IF NOT EXISTS idx_kg_nodes_type
            ON kg_nodes (type);

        CREATE INDEX IF NOT EXISTS idx_kg_nodes_importance
            ON kg_nodes (importance DESC);
        """
    )


def _ensure_kg_edges_schema(conn: sqlite3.Connection) -> None:
    """
    kg_edges — directed relationships between kg_nodes.

    DESIGN DECISIONS:
      Directed edges only. "Seven uses PostgreSQL" is not the same as
      "PostgreSQL uses Seven" — direction matters semantically. Undirected
      relationships (related_to, similar_to) are stored as a single directed
      edge by convention: lower node id → higher node id, so there is never
      a duplicate pair.

      relation is a controlled vocabulary defined in KnowledgeGraph/constants.py
      (Phase 2). Examples: uses, built_with, depends_on, contains, created,
      replaced, part_of, related_to, located_in, knows, worked_on, prefers,
      learned, mentions, contradicts.
      The validator (Phase 2) rejects any relation not in the allowed set.

      active flag: when a relationship changes (migrated from SQLite to
      PostgreSQL), deactivate the old edge rather than deleting it. This
      preserves historical graph state and keeps evidence_memory_ids intact
      for explainability ("why did you think I used SQLite?").

      evidence_memory_ids: JSON list of episodic/semantic memory ids that
      support this relationship. Critical for explainability — Seven can
      always trace a graph claim back to the raw memories behind it.

    COLUMNS:
      id                  — UUID primary key.
      source_id           — FK to kg_nodes.id (the "from" node).
      target_id           — FK to kg_nodes.id (the "to" node).
      relation            — controlled relationship type string.
      confidence          — 0.0-1.0. Increases when additional memories
                            confirm the relationship.
      weight              — traversal priority. Computed as
                            relation_strength × confidence × relevance.
                            Higher weight = preferred during BFS/DFS.
      active              — 1 = current, 0 = historical/superseded.
      evidence_memory_ids — JSON list of memory ids supporting this edge.
      created_at          — ISO timestamp.
      updated_at          — ISO timestamp.

    INDEXES:
      idx_kg_edges_source  — all outgoing edges from a node. Primary traversal path.
      idx_kg_edges_target  — all incoming edges to a node. Reverse traversal.
      idx_kg_edges_pair    — (source_id, target_id) pair lookup. Used to check
                             whether an edge already exists before inserting.
      idx_kg_edges_active  — filter active-only edges without a full scan.

    FOREIGN KEYS:
      Both source_id and target_id reference kg_nodes(id) with CASCADE DELETE.
      If a node is deleted, all its edges are automatically removed — no orphaned
      edges. This is safe because node deletion is always an intentional operation
      logged in kg_graph_logs before execution.
    """
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS kg_edges (
            id                   TEXT PRIMARY KEY,
            source_id            TEXT NOT NULL REFERENCES kg_nodes(id) ON DELETE CASCADE,
            target_id            TEXT NOT NULL REFERENCES kg_nodes(id) ON DELETE CASCADE,
            relation             TEXT NOT NULL,
            confidence           REAL NOT NULL DEFAULT 0.5,
            weight               REAL NOT NULL DEFAULT 0.5,
            active               INTEGER NOT NULL DEFAULT 1,
            evidence_memory_ids  TEXT NOT NULL DEFAULT '[]',  -- JSON list
            created_at           TEXT NOT NULL,
            updated_at           TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_kg_edges_source
            ON kg_edges (source_id, active);

        CREATE INDEX IF NOT EXISTS idx_kg_edges_target
            ON kg_edges (target_id, active);

        CREATE INDEX IF NOT EXISTS idx_kg_edges_pair
            ON kg_edges (source_id, target_id, active);

        CREATE INDEX IF NOT EXISTS idx_kg_edges_active
            ON kg_edges (active);
        """
    )


def _ensure_kg_node_aliases_schema(conn: sqlite3.Connection) -> None:
    """
    kg_node_aliases — alternative names that all resolve to the same node.

    PURPOSE:
      The primary entity resolution strategy during Sleep Mode and query-time
      retrieval. Before touching the embedding index (expensive), the pipeline
      checks: does this string exactly match any alias? If yes, the node is
      found in O(1). This covers "postgres", "pgsql", "Postgres DB" all mapping
      to the canonical "PostgreSQL" node.

      Aliases are case-insensitively stored (lowercased at write time by
      kg_db_client) so lookups are always case-insensitive without collation
      complexity.

    COLUMNS:
      id         — UUID primary key.
      node_id    — FK to kg_nodes.id. CASCADE DELETE so aliases are cleaned
                   up automatically when a node is removed.
      alias      — the alternative name, stored lowercase.
      created_at — ISO timestamp.

    INDEXES:
      idx_kg_aliases_alias   — exact alias lookup. O(1). The hot path.
      idx_kg_aliases_node_id — all aliases for a given node (for display/audit).

    UNIQUE constraint on (node_id, alias) prevents duplicate aliases on the
    same node (the same alias on different nodes IS allowed — entity resolution
    handles ambiguity).
    """
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS kg_node_aliases (
            id         TEXT PRIMARY KEY,
            node_id    TEXT NOT NULL REFERENCES kg_nodes(id) ON DELETE CASCADE,
            alias      TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE (node_id, alias)
        );

        CREATE INDEX IF NOT EXISTS idx_kg_aliases_alias
            ON kg_node_aliases (alias);

        CREATE INDEX IF NOT EXISTS idx_kg_aliases_node_id
            ON kg_node_aliases (node_id);
        """
    )


def _ensure_kg_node_keywords_schema(conn: sqlite3.Connection) -> None:
    """
    kg_node_keywords — inverted keyword index over node names and headings.

    PURPOSE:
      When neither the name index nor the alias index returns a match (the
      query uses a word that appears in a node's description but not its
      canonical name or aliases), the keyword index is the next fallback
      before the embedding search.

      Example: query "relational database" → keyword "relational" maps to
      the PostgreSQL node via its heading "Relational Database System", even
      though "relational" is not an alias.

      Keywords are individual meaningful words extracted from node name +
      heading at insert/update time by kg_db_client (stopwords stripped,
      lowercased). Compound phrases are not stored as single keywords —
      individual words are, so multi-word queries accumulate candidate sets
      via OR across their component words.

    COLUMNS:
      id         — UUID primary key.
      node_id    — FK to kg_nodes.id. CASCADE DELETE.
      keyword    — single lowercase word.
      created_at — ISO timestamp.

    INDEXES:
      idx_kg_keywords_keyword — the hot lookup path: keyword → [node_ids].
      idx_kg_keywords_node_id — all keywords for a node (for cleanup on update).

    UNIQUE on (node_id, keyword) prevents the same word being indexed twice
    for the same node if name and heading share a word.
    """
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS kg_node_keywords (
            id         TEXT PRIMARY KEY,
            node_id    TEXT NOT NULL REFERENCES kg_nodes(id) ON DELETE CASCADE,
            keyword    TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE (node_id, keyword)
        );

        CREATE INDEX IF NOT EXISTS idx_kg_keywords_keyword
            ON kg_node_keywords (keyword);

        CREATE INDEX IF NOT EXISTS idx_kg_keywords_node_id
            ON kg_node_keywords (node_id);
        """
    )


def _ensure_kg_memory_nodes_schema(conn: sqlite3.Connection) -> None:
    """
    kg_memory_nodes — link table between memories and graph nodes.

    PURPOSE:
      Every graph node and edge is ultimately supported by memories. This
      table makes that relationship explicit and bidirectional:
        - Given a node: which memories mention this concept?
        - Given a memory: which nodes did we extract from it?

      This is what allows Seven to answer "why do you think I use PostgreSQL?"
      by traversing memory_id → episodic/semantic store → raw text.

      The column `entities_mentioned` in episodic_memory_store.py was
      explicitly RESERVED for this purpose — this table is the materialisation
      of that reservation.

    COLUMNS:
      id          — UUID primary key.
      node_id     — FK to kg_nodes.id. CASCADE DELETE.
      memory_id   — the ChromaDB document id from episodic_memory or
                    semantic_memory. NOT a FK (different DB) — integrity
                    is maintained at the application level.
      memory_type — 'episodic' | 'semantic'. Tells Phase 3 retrieval which
                    ChromaDB collection to query when fetching the raw memory.
      relevance   — 0.0-1.0. How central is this concept to this memory?
                    A memory entirely about PostgreSQL has relevance 1.0;
                    one that mentions it in passing has relevance 0.3.
      created_at  — ISO timestamp.

    INDEXES:
      idx_kg_memory_nodes_node_id   — all memories for a node. Used by
                                      Phase 3 to fetch evidence for a concept.
      idx_kg_memory_nodes_memory_id — all nodes extracted from a memory. Used
                                      during Sleep Mode to skip memories that
                                      have already been fully processed.

    UNIQUE on (node_id, memory_id) prevents the same memory being linked to
    the same node twice (can happen if a memory appears in two sleep batches).
    """
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS kg_memory_nodes (
            id          TEXT PRIMARY KEY,
            node_id     TEXT NOT NULL REFERENCES kg_nodes(id) ON DELETE CASCADE,
            memory_id   TEXT NOT NULL,
            memory_type TEXT NOT NULL DEFAULT 'episodic',
            relevance   REAL NOT NULL DEFAULT 0.5,
            created_at  TEXT NOT NULL,
            UNIQUE (node_id, memory_id)
        );

        CREATE INDEX IF NOT EXISTS idx_kg_memory_nodes_node_id
            ON kg_memory_nodes (node_id);

        CREATE INDEX IF NOT EXISTS idx_kg_memory_nodes_memory_id
            ON kg_memory_nodes (memory_id);
        """
    )


def _ensure_kg_graph_logs_schema(conn: sqlite3.Connection) -> None:
    """
    kg_graph_logs — append-only audit log of every graph operation.

    PURPOSE:
      Version history without a full diff engine. Every mutation to the
      knowledge graph (insert_node, insert_edge, merge_node, deactivate_edge,
      update_confidence, etc.) is recorded here BEFORE the mutation is applied.
      If a batch goes wrong, the log shows exactly what happened and in what order.

      This is also the foundation for a future "undo last sleep batch" command:
      replay the log in reverse to restore the previous state.

    COLUMNS:
      id          — UUID primary key.
      operation   — string name of the operation: insert_node | insert_edge |
                    update_node | update_edge | deactivate_edge | merge_node |
                    delete_node | link_memory.
      entity_type — 'node' | 'edge' | 'alias' | 'keyword' | 'memory_link'.
      entity_id   — the id of the affected node or edge.
      details     — JSON blob with the full before/after state or the parameters
                    used. Enough to reconstruct what changed without a separate
                    snapshot table.
      source      — who triggered the operation: 'sleep_pipeline' | 'manual' |
                    'validator' | 'migration'.
      created_at  — ISO timestamp. The log is ordered by this column.

    INDEXES:
      idx_kg_logs_entity_id  — all log entries for a specific node or edge.
                               Used during audit ("what happened to this node?").
      idx_kg_logs_created_at — chronological scan for "show me what the last
                               sleep batch did".
      idx_kg_logs_operation  — filter by operation type for analytics.

    NO DELETE, NO UPDATE. This table is append-only by contract. The application
    layer enforces this — there is no delete_log_entry() function in kg_db_client.
    """
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS kg_graph_logs (
            id          TEXT PRIMARY KEY,
            operation   TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            entity_id   TEXT NOT NULL,
            details     TEXT NOT NULL DEFAULT '{}',  -- JSON
            source      TEXT NOT NULL DEFAULT 'sleep_pipeline',
            created_at  TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_kg_logs_entity_id
            ON kg_graph_logs (entity_id);

        CREATE INDEX IF NOT EXISTS idx_kg_logs_created_at
            ON kg_graph_logs (created_at);

        CREATE INDEX IF NOT EXISTS idx_kg_logs_operation
            ON kg_graph_logs (operation);
        """
    )


def _ensure_kg_sleep_queue_schema(conn: sqlite3.Connection) -> None:
    """
    kg_sleep_queue — durable input queue for the Knowledge Graph sleep
    pipeline, one row per session awaiting processing.

    PURPOSE:
      A session's raw context (full conversation, related semantic
      memory ids) lives in active_sessions only while the session is
      open — close_session() deletes that row entirely once the
      episodic write succeeds. Without this table, anything the sleep
      pipeline needs beyond the episode's title+summary would already
      be gone by the time it runs. This table is written BEFORE
      close_session() runs, so it becomes the sleep pipeline's durable,
      crash-safe input — independent of ChromaDB and independent of
      active_sessions' lifecycle.

    COLUMNS:
      session_id           — PRIMARY KEY. Same session_id used across
                              active_sessions and episodic_memory
                              metadata — the join key for everything.
      episodic_memory_id    — the id returned by
                              episodic_memory_store.insert_episode() for
                              this session. Lets the pipeline fetch the
                              full episode (title + summary) from
                              ChromaDB without re-deriving it.
      semantic_memory_ids   — JSON list of ChromaDB semantic_memory ids
                              created during this session (from
                              session_memory_tracker.get_and_clear()).
                              Lets the pipeline fetch each fact's text
                              individually via semantic_memory.get_by_ids().
      conversation_text     — chunk_summaries joined into one string, or
                              a full_conversation snippet fallback when
                              no chunk summary exists yet. Gives the
                              sleep pipeline's entity extractor real
                              narrative context instead of just the
                              compressed episode summary.
      queued_at             — ISO timestamp, when this row was written.
                              Sleep pipeline processes oldest-first.
      processed_at          — NULL while pending. Set to an ISO
                              timestamp once the sleep pipeline finishes
                              extracting entities/edges from this
                              session. NULL rows are exactly the sleep
                              pipeline's backlog — COUNT(*) WHERE
                              processed_at IS NULL is "sessions
                              remaining to process".

    INDEXES:
      idx_kg_sleep_queue_processed_at — lets get_pending_sessions() and
                                        count_pending() filter
                                        WHERE processed_at IS NULL
                                        without a full table scan.

    Rows are not deleted on processing (processed_at is stamped
    instead) so a completed sleep batch stays inspectable; a separate
    delete_processed(older_than_days) cleanup call is used to prune old
    processed rows once they're no longer useful for debugging.
    """
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS kg_sleep_queue (
            session_id           TEXT PRIMARY KEY,
            episodic_memory_id   TEXT NOT NULL,
            semantic_memory_ids  TEXT NOT NULL DEFAULT '[]',  -- JSON list
            conversation_text    TEXT NOT NULL DEFAULT '',
            queued_at            TEXT NOT NULL,
            processed_at         TEXT  -- NULL = unprocessed
        );

        CREATE INDEX IF NOT EXISTS idx_kg_sleep_queue_processed_at
            ON kg_sleep_queue (processed_at);
        """
    )


def _ensure_schema() -> None:
    global _schema_ready
    if _schema_ready:
        return
    with _init_lock:
        if _schema_ready:
            return
        conn = _create_connection()
        try:
            for name, fn in (
                ("working_memory",   _ensure_working_memory_schema),
                ("active_sessions",  _ensure_active_sessions_schema),
                ("kg_nodes",         _ensure_kg_nodes_schema),
                ("kg_edges",         _ensure_kg_edges_schema),
                ("kg_node_aliases",  _ensure_kg_node_aliases_schema),
                ("kg_node_keywords", _ensure_kg_node_keywords_schema),
                ("kg_memory_nodes",  _ensure_kg_memory_nodes_schema),
                ("kg_graph_logs",    _ensure_kg_graph_logs_schema),
                ("kg_sleep_queue",   _ensure_kg_sleep_queue_schema),
            ):
                try:
                    fn(conn)
                    conn.commit()
                    log.info("Local SQLite schema ready for '%s' at %s", name, DB_PATH)
                except Exception:
                    conn.rollback()
                    log.exception("Failed to initialize schema for '%s' (non-fatal, other tables continue).", name)
        finally:
            conn.close()
        _schema_ready = True


_ensure_schema()