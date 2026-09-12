"""
Database/kg_db_client.py

Facade that re-exports the entire Knowledge Graph DB layer from one import.

Phase 2 (sleep pipeline) and Phase 3 (retrieval) import from here:
    from Database import kg_db_client
    kg_db_client.insert_node(...)
    kg_db_client.insert_edge(...)

The actual implementations live in:
    kg_constants.py     — shared constants (NODE_TYPES, RELATION_TYPES, etc.)
    kg_node_client.py   — node CRUD + keyword index maintenance
    kg_alias_client.py  — alias + keyword index functions
    kg_edge_client.py   — edge CRUD
    kg_link_client.py   — memory-node links + graph audit log
"""

## noqa: F401 is a comment understood by Python linting tools such as Flake8.
# So that we can re-export functions from other modules without triggering a linting warning, we add this comment to the import lines below.
# This was a new thing which i understood today

# Node CRUD
from Database.kg_node_client import (          # noqa: F401
    insert_node,
    get_node_by_id,
    get_node_by_name,
    get_nodes_by_type,
    update_node,
    delete_node,
    increment_access_count,
    increment_traversal_count,
    search_nodes_by_name_prefix,
    _extract_keywords,
)

# Alias + keyword index
from Database.kg_alias_client import (         # noqa: F401
    add_alias,
    get_aliases_for_node,
    get_node_by_alias,
    delete_alias,
    add_keyword,
    get_nodes_by_keyword,
    get_keywords_for_node,
    delete_keywords_for_node,
    _index_node_keywords,
)

# Edge CRUD
from Database.kg_edge_client import (          # noqa: F401
    insert_edge,
    get_edge_by_id,
    get_edges_from,
    get_edges_to,
    get_edge_between,
    update_edge_confidence,
    add_evidence_to_edge,
    deactivate_edge,
    reactivate_edge,
    delete_edge,
    RELATION_TYPES,
)

# Memory-node links + graph log
from Database.kg_link_client import (          # noqa: F401
    link_memory_to_node,
    get_memory_links_for_node,
    get_nodes_for_memory,
    unlink_memory_from_node,
    get_unprocessed_memory_ids,
    log_graph_operation,
    get_log_entries_for_entity,
    get_recent_log_entries,
    get_log_entry_count,
)

# Shared constants
from Database.kg_constants import (            # noqa: F401
    NODE_TYPES,
    RELATION_TYPES,  # re-exported for convenience
    _STOPWORDS,
)