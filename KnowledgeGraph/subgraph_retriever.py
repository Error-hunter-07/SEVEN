from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
import Database.kg_db_client as kg
from GlobalHelpers.logger import get_logger
from KnowledgeGraph.constants import SUBGRAPH_MAX_HOPS, SUBGRAPH_MAX_NODES, SUBGRAPH_MAX_EDGES_PER_NODE
log = get_logger(__name__)

@dataclass
class Subgraph:
    nodes: dict = field(default_factory=dict)
    edges: list = field(default_factory=list)
    seed_ids: set = field(default_factory=set)
    new_ids: set = field(default_factory=set)
    text: str = ""

def _bfs(seed_ids, max_hops, max_nodes):
    visited_nodes = {}; visited_edges = {}; frontier = deque(); seen = set()
    for nid in seed_ids:
        if nid and nid not in seen: frontier.append(nid); seen.add(nid)
    for hop in range(max_hops + 1):
        if not frontier: break
        next_frontier = []
        while frontier:
            nid = frontier.popleft()
            if nid not in visited_nodes:
                node = kg.get_node_by_id(nid)
                if not node: continue
                visited_nodes[nid] = node
                if len(visited_nodes) >= max_nodes: continue
            out = kg.get_edges_from(nid, active_only=True, limit=SUBGRAPH_MAX_EDGES_PER_NODE)
            inc = kg.get_edges_to(nid,   active_only=True, limit=SUBGRAPH_MAX_EDGES_PER_NODE)
            for edge in out + inc:
                if edge["id"] not in visited_edges: visited_edges[edge["id"]] = edge
                if hop < max_hops and len(visited_nodes) < max_nodes:
                    other = edge["target_id"] if edge["source_id"] == nid else edge["source_id"]
                    if other not in seen: next_frontier.append(other); seen.add(other)
        if len(visited_nodes) < max_nodes: frontier = deque(next_frontier)
    return visited_nodes, list(visited_edges.values())

def _format_text(nodes, edges, seed_ids, new_ids):
    if not nodes and not edges: return ""
    lines = ["Seed nodes (resolved this batch):"]
    for nid in seed_ids:
        node = nodes.get(nid)
        if node:
            flag = " [NEW]" if nid in new_ids else ""
            lines.append(f"  {node['name']} ({nid[:8]}) type={node['type']} conf={node['confidence']:.2f}{flag}")
    if edges:
        lines.append(""); lines.append("Active edges:")
        for edge in sorted(edges, key=lambda e: -e["weight"]):
            src = nodes.get(edge["source_id"]); tgt = nodes.get(edge["target_id"])
            sl = f"{src['name']} ({edge['source_id'][:8]})" if src else edge["source_id"][:8]
            tl = f"{tgt['name']} ({edge['target_id'][:8]})" if tgt else edge["target_id"][:8]
            lines.append(f"  [{edge['id'][:8]}]  {sl} --{edge['relation']}--> {tl}  conf={edge['confidence']:.2f}")
    return "\n".join(lines)

def fetch_subgraph(resolved):
    valid = [r for r in resolved if r.node_id]
    if not valid: return Subgraph()
    seed_ids = {r.node_id for r in valid}; new_ids = {r.node_id for r in valid if r.is_new}
    nodes, edges = _bfs(seed_ids, SUBGRAPH_MAX_HOPS, SUBGRAPH_MAX_NODES)
    text = _format_text(nodes, edges, seed_ids, new_ids)
    return Subgraph(nodes=nodes, edges=edges, seed_ids=seed_ids, new_ids=new_ids, text=text)