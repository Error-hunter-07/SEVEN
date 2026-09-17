"""
KnowledgeGraph/subgraph_retriever.py

STEP 3 of the sleep pipeline: fetch the local subgraph around resolved
entities to give the operation proposer context about what already exists
in the Knowledge Graph.

POSITION IN PIPELINE:
  memory_selector  →  entity_extractor  →  entity_resolver  →  [subgraph_retriever]
                                                                        ↓
                                                                operation_proposer

UNIT OF WORK: list[ResolutionResult] (resolved entities from one session)
  Receives the resolved entities with their graph node IDs. Starting from
  those seed nodes, traverses the graph using BFS up to SUBGRAPH_MAX_HOPS
  hops, collecting up to SUBGRAPH_MAX_NODES nodes and
  SUBGRAPH_MAX_EDGES_PER_NODE edges per node.

WHY THIS MATTERS:
  The operation proposer needs to know what edges already exist so it can:
  - Avoid inserting duplicate edges (use update_edge_confidence instead)
  - Decide whether to deactivate superseded edges
  - Understand the entity's neighbourhood to make informed proposals

  Without subgraph context, the proposer would be blind to existing
  relationships and could create contradictory or redundant edges.

OUTPUT:
  Subgraph dataclass containing:
    - nodes: dict[node_id → node_dict]
    - edges: list[edge_dict]
    - seed_ids: the original resolved node ids
    - new_ids: subset of seed_ids that were just created
    - text: formatted text representation for the LLM prompt
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import Database.kg_db_client as kg
from GlobalHelpers.logger import get_logger
from KnowledgeGraph.constants import (
    SUBGRAPH_MAX_HOPS,
    SUBGRAPH_MAX_NODES,
    SUBGRAPH_MAX_EDGES_PER_NODE,
)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Output data structure
# ---------------------------------------------------------------------------

@dataclass
class Subgraph:
    """
    The local subgraph around resolved entities.

    nodes:     dict mapping node_id → node dict (name, type, confidence, etc.)
    edges:     list of edge dicts (source_id, target_id, relation, weight, etc.)
    seed_ids:  set of resolved entity node ids (the BFS starting points)
    new_ids:   subset of seed_ids that were just created in this pipeline run
    text:      formatted text for the operation proposer LLM prompt
    """
    nodes: dict = field(default_factory=dict)
    edges: list = field(default_factory=list)
    seed_ids: set = field(default_factory=set)
    new_ids: set = field(default_factory=set)
    text: str = ""


# Internal: BFS traversal

def _bfs(seed_ids, max_hops, max_nodes):
    """
    Breadth-first search from seed nodes, collecting nodes and edges.

    Traverses outgoing and incoming edges from each node, up to max_hops
    levels deep. Stops collecting nodes when max_nodes is reached (highest-
    importance nodes are not prioritized — simple insertion order is used
    since the graph is small locally).

    Returns (visited_nodes: dict, visited_edges: list).
    """
    visited_nodes = {}
    visited_edges = {}
    frontier = deque()
    seen = set()

    # Seed the frontier with resolved entity node ids
    for nid in seed_ids:
        if nid and nid not in seen:
            frontier.append(nid)
            seen.add(nid)

    for hop in range(max_hops + 1):
        if not frontier:
            break

        next_frontier = []
        while frontier:
            nid = frontier.popleft()

            # Visit node (skip if already visited or not in graph)
            if nid not in visited_nodes:
                node = kg.get_node_by_id(nid)
                if not node:
                    continue
                visited_nodes[nid] = node
                if len(visited_nodes) >= max_nodes:
                    continue

            # Collect edges in both directions
            out = kg.get_edges_from(nid, active_only=True, limit=SUBGRAPH_MAX_EDGES_PER_NODE)
            inc = kg.get_edges_to(nid,   active_only=True, limit=SUBGRAPH_MAX_EDGES_PER_NODE)

            for edge in out + inc:
                if edge["id"] not in visited_edges:
                    visited_edges[edge["id"]] = edge

                # Enqueue the other endpoint for next hop (if within limits)
                if hop < max_hops and len(visited_nodes) < max_nodes:
                    other = (
                        edge["target_id"]
                        if edge["source_id"] == nid
                        else edge["source_id"]
                    )
                    if other not in seen:
                        next_frontier.append(other)
                        seen.add(other)

        if len(visited_nodes) < max_nodes:
            frontier = deque(next_frontier)

    return visited_nodes, list(visited_edges.values())


# Internal: format subgraph as text for LLM prompt


def _format_text(nodes, edges, seed_ids, new_ids):
    """
    Format the subgraph as human-readable text for the operation proposer.

    Sections:
      1. Seed nodes — the resolved entities (marked [NEW] if just created)
      2. Active edges — sorted by weight descending (highest importance first)

    Returns empty string if no nodes or edges were found.
    """
    if not nodes and not edges:
        return ""

    lines = ["Seed nodes (resolved this batch):"]
    for nid in seed_ids:
        node = nodes.get(nid)
        if node:
            flag = " [NEW]" if nid in new_ids else ""
            lines.append(
                f"  {node['name']} ({nid}) type={node['type']} "
                f"conf={node['confidence']:.2f}{flag}"
            )

    if edges:
        lines.append("")
        lines.append("Active edges:")
        for edge in sorted(edges, key=lambda e: -e["weight"]):
            src = nodes.get(edge["source_id"])
            tgt = nodes.get(edge["target_id"])
            sl = f"{src['name']} ({edge['source_id']})" if src else edge["source_id"]
            tl = f"{tgt['name']} ({edge['target_id']})" if tgt else edge["target_id"]
            lines.append(
                f"  [{edge['id']}]  {sl} --{edge['relation']}--> {tl}  "
                f"conf={edge['confidence']:.2f}"
            )

    return "\n".join(lines)


# Public API

def fetch_subgraph(resolved):
    """
    Fetch the local subgraph around resolved entities.

    Traverses from the resolved node ids using BFS, collecting the
    surrounding nodes and edges. Returns a Subgraph with both the
    structured data and a formatted text representation.

    Args:
      resolved: list[ResolutionResult] from entity_resolver.

    Returns:
      Subgraph — may have empty nodes/edges if no resolved entities
      have node_ids or if the graph has no connections around them.
    """
    valid = [r for r in resolved if r.node_id]
    if not valid:
        return Subgraph()

    seed_ids = {r.node_id for r in valid}
    new_ids = {r.node_id for r in valid if r.is_new}

    nodes, edges = _bfs(seed_ids, SUBGRAPH_MAX_HOPS, SUBGRAPH_MAX_NODES)
    text = _format_text(nodes, edges, seed_ids, new_ids)

    return Subgraph(
        nodes=nodes,
        edges=edges,
        seed_ids=seed_ids,
        new_ids=new_ids,
        text=text,
    )
