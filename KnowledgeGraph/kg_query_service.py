"""
KnowledgeGraph/kg_query_service.py

Phase 3 retrieval orchestrator for the Knowledge Graph.

Public entry point: query_knowledge_graph(query, top_n=5) -> str
  Runs four candidate-generation methods, fuses them into a ranked seed-node
  list, reinforces the graph (traversal counts), and hands the seed ids to
  subgraph_retriever.fetch_subgraph() to build the final context text.

Tier model
----------
Tier 1 — exact identity match: the query text IS the node's canonical name
         (case-insensitive) or a registered alias. No ambiguity about which
         entity was meant. Skips RRF entirely, ordered only by graph_prior
         to break ties among multiple tier-1 hits.
Tier 2 — everything else (name-prefix, keyword, semantic hits). Ranked by
         Reciprocal Rank Fusion blended with graph_prior, since for fuzzy
         matches "how many independent methods agree" is real evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import Database.kg_node_client as kg_node_client
import Database.kg_alias_client as kg_alias_client
from GlobalHelpers.logger import get_logger

log = get_logger(__name__)

# Standard RRF damping constant.
_RRF_K = 60

_NO_MATCH_TEXT = (
    "No relevant knowledge graph entries were found for this query."
)


# ---------------------------------------------------------------------------
# Query tokenization
# ---------------------------------------------------------------------------

def _extract_names(query: str) -> list[str]:
    # _extract_keywords(name, heading) joins both args and tokenizes.
    return kg_node_client._extract_keywords(query, "")


# ---------------------------------------------------------------------------
# Candidate generation — one function per search method
# ---------------------------------------------------------------------------

def _search_exact_name(query: str) -> Tuple[set, list]:
    """
    Returns (tier1, tier2):
      tier1 — node ids whose canonical name exactly matches a query token
              (case-insensitive) — verified identity match.
      tier2 — node ids that only prefix-matched — feeds the RRF pool.
    """
    keywords = _extract_names(query)
    tier1, tier2, seen = set(), [], set()
    for kw in keywords:
        if len(kw) < 3:
            continue
        for node in kg_node_client.search_nodes_by_name_prefix(kw, limit=5):
            if node["name"].lower() == kw.lower():
                tier1.add(node["id"])
            elif node["id"] not in seen:
                tier2.append(node["id"])
                seen.add(node["id"])
    return tier1, tier2


def _search_alias(query: str) -> list:
    """Exact alias lookup per token — also tier-1 confidence (a hit means
    this string is a registered alternate name for the node)."""
    words = _extract_names(query)
    tier1 = []
    for w in words:
        node = kg_alias_client.get_node_by_alias(w)
        if node is None:
            continue
        tier1.append(node["id"])
    return tier1


def _search_keyword(query: str) -> list:
    """Inverted keyword index lookup per token — tier-2 (matches the
    node's indexed text, not its identity)."""
    words = _extract_names(query)
    tier2 = []
    for w in words:
        nodes = kg_alias_client.get_nodes_by_keyword(w)
        tier2.extend(node["id"] for node in nodes)
    return tier2


def _search_semantic(query: str, k: int = 5) -> list[str]:
    """Doubly-indirect tier-2 source: embedding similarity finds a memory,
    then the memory->node link table surfaces the node."""
    from MemoryManagement.semantic_memory.semantic_memory import semantic_memory
    import MemoryManagement.episodic_memory.episodic_memory_store as episodic_memory_store
    import Database.kg_link_client as kg_link_client

    node_ids = []
    for hit in semantic_memory.retrieve(query, k=k, update_access=True):
        node_ids += [n["id"] for n in kg_link_client.get_nodes_for_memory(hit["id"])]

    for hit in episodic_memory_store.search_episodes(query, k=k):
        node_ids += [n["id"] for n in kg_link_client.get_nodes_for_memory(hit["id"])]
        episodic_memory_store.mark_recalled(hit["id"])

    return list(dict.fromkeys(node_ids))  # dedupe, order-preserving


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _graph_prior(node: dict | None) -> float:
    """Structural confidence score, independent of query relevance."""
    if not node:
        return 0.0

    importance = node.get("importance") or 0.0
    confidence = node.get("confidence") or 0.0
    access_count = node.get("access_count") or 0
    traversal_count = node.get("traversal_count") or 0

    activity_component = math.log(1 + access_count + traversal_count) / math.log(51)
    return 0.4 * importance + 0.3 * confidence + 0.3 * min(1.0, activity_component)


def _merge_and_rank(
    method_hits: dict[str, list[str]],
    nodes_by_id: dict[str, dict],
) -> dict[str, float]:
    """
    RRF + graph_prior blend over the TIER-2 pool only. Callers are
    responsible for excluding tier-1 (exact) ids from `method_hits` before
    calling this — tier-1 handling lives in select_seed_nodes(), not here.

    Uses the pre-fetched `nodes_by_id` cache instead of re-querying the DB
    per candidate.
    """
    if not any(method_hits.values()):
        return {}

    ranks: dict[str, float] = {}
    for method, hits in method_hits.items():
        for position, node_id in enumerate(hits, start=1):
            ranks.setdefault(node_id, 0.0)
            ranks[node_id] += 1.0 / (_RRF_K + position)

    max_val = max(ranks.values())
    min_val = min(ranks.values())
    if max_val > min_val:
        ranks = {k: (v - min_val) / (max_val - min_val) for k, v in ranks.items()}
    else:
        ranks = {k: 1.0 for k in ranks}  # every candidate tied

    final_scores: dict[str, float] = {}
    for node_id, rank in ranks.items():
        node = nodes_by_id.get(node_id)
        if node is None:
            continue  # candidate vanished between search and scoring
        final_scores[node_id] = 0.75 * rank + 0.25 * _graph_prior(node)

    return final_scores


# ---------------------------------------------------------------------------
# Selection — owns tier ordering, exact-id exclusion, batched fetch,
# truncation, and graph reinforcement
# ---------------------------------------------------------------------------

def _fetch_nodes_by_id(ids: set[str]) -> dict[str, dict]:
    """Single pass fetching each unique candidate node exactly once."""
    out = {}
    for node_id in ids:
        node = kg_node_client.get_node_by_id(node_id)
        if node is not None:
            out[node_id] = node
    return out


def select_seed_nodes(
    query: str,
    top_n: int = 5,
    reserve_semantic_slots: int = 3,
) -> list[dict]:
    """
    Runs all four search methods, merges + ranks them, and returns the
    top_n seed node dicts (tier-1 exact matches first, sorted by
    graph_prior; then tier-2 RRF+graph_prior matches).

    reserve_semantic_slots: if >0 and tier-1 alone would fill every slot,
    reserve this many slots for the best tier-2 candidates instead —
    defaults to off (tier-1 exact matches consume every slot: if the user
    named specific entities, search those).

    Reinforcement: selected nodes get increment_traversal_count() bumped.
    """
    exact_ids, exact_prefix_hits = _search_exact_name(query)
    alias_ids = _search_alias(query)
    keyword_ids = _search_keyword(query)
    semantic_ids = _search_semantic(query, k=top_n)

    # Alias hits are also exact identity matches — fold into tier-1.
    exact_ids = exact_ids | set(alias_ids)

    # Exclude tier-1 ids from the tier-2 pools so they don't consume RRF
    # rank positions or get double-seeded.
    method_hits = {
        "exact_prefix": [i for i in exact_prefix_hits if i not in exact_ids],
        "alias": [i for i in alias_ids if i not in exact_ids],
        "keyword": [i for i in keyword_ids if i not in exact_ids],
        "semantic": [i for i in semantic_ids if i not in exact_ids],
    }

    all_ids = set(exact_ids)
    for hits in method_hits.values():
        all_ids.update(hits)
    nodes_by_id = _fetch_nodes_by_id(all_ids)

    tier1_sorted = sorted(
        exact_ids, key=lambda i: _graph_prior(nodes_by_id.get(i)), reverse=True
    )
    tier2_scores = _merge_and_rank(method_hits, nodes_by_id)
    tier2_sorted = sorted(tier2_scores, key=tier2_scores.get, reverse=True)

    if reserve_semantic_slots > 0 and len(tier1_sorted) >= top_n:
        keep_tier1 = max(top_n - reserve_semantic_slots, 0)
        seed_ids = tier1_sorted[:keep_tier1] + tier2_sorted
    else:
        seed_ids = tier1_sorted + tier2_sorted

    seed_ids = seed_ids[:top_n]

    for node_id in seed_ids:
        kg_node_client.increment_traversal_count(node_id)

    results = []
    for node_id in seed_ids:
        node = nodes_by_id.get(node_id)
        if node is None:
            continue
        node["_relevance_score"] = tier2_scores.get(node_id, 1.0)  # tier-1 = 1.0
        results.append(node)
    return results


# ---------------------------------------------------------------------------
# Context text building
# ---------------------------------------------------------------------------

@dataclass
class _SeedResolution:
    """
    Minimal duck-typed stand-in for entity_resolver.ResolutionResult.

    subgraph_retriever.fetch_subgraph() only ever reads `.node_id` and
    `.is_new` off each item in its `resolved` argument (confirmed by
    reading fetch_subgraph's body: `r.node_id`, `r.is_new`, nothing else).
    Rather than importing and guessing at ResolutionResult's full
    constructor from entity_resolver.py (which likely carries extra
    fields like confidence/name that a query-time seed has no value for),
    this local stand-in satisfies exactly the interface fetch_subgraph
    actually consumes.
    """
    node_id: str
    is_new: bool = False


def _build_context_text(seed_node_ids: list[str]) -> str:
    from KnowledgeGraph.subgraph_retriever import fetch_subgraph

    resolutions = [_SeedResolution(node_id=nid, is_new=False) for nid in seed_node_ids]
    subgraph = fetch_subgraph(resolutions)
    # fetch_subgraph/._format_text returns "" if no nodes or edges were
    # found at all — fall back to the fixed no-match string in that case.
    return subgraph.text or _no_match_text()


def _no_match_text() -> str:
    return _NO_MATCH_TEXT


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def query_knowledge_graph(query: str, top_n: int = 5) -> str:
    """LLM-facing entry point. Returns formatted context text, or the
    fixed no-match string if nothing relevant was found."""
    if not query or not query.strip():
        log.warning("query_knowledge_graph: called with empty query.")
        return _no_match_text()

    seed_nodes = select_seed_nodes(query, top_n=top_n)
    if not seed_nodes:
        return _no_match_text()

    return _build_context_text([n["id"] for n in seed_nodes])