"""
KnowledgeGraph/constants.py

All configuration for the Knowledge Graph sleep pipeline in one place.

WHAT LIVES HERE vs Database/kg_constants.py:
  Database/kg_constants.py — DB-layer constants: NODE_TYPES, RELATION_TYPES,
      _STOPWORDS, _now(), _new_id(). Only the DB client files import from there.

  This file — pipeline-layer constants: everything the sleep pipeline
      (memory_selector, entity_extractor, entity_resolver, subgraph_retriever,
      operation_proposer, validator, sleep_scheduler) needs.
      Imports NODE_TYPES and RELATION_TYPES from the DB layer and re-exports
      them so pipeline code has exactly one import point for all constants.

TUNING:
  All numeric thresholds and batch sizes are defined here. To tune the
  pipeline's behaviour — make it more aggressive, more conservative, process
  larger batches, require higher confidence before writing — change values
  here. No logic files need to be touched.

PROMPTS:
  Both LLM prompt templates live here so they can be reviewed, tested, and
  iterated on independently of the pipeline code that calls them.
  The entity extraction prompt (ENTITY_EXTRACTION_PROMPT) and the operation
  proposal prompt (OPERATION_PROPOSAL_PROMPT) are the two places where the
  quality of the Knowledge Graph is determined. Poor prompts = poor graph.
"""

from __future__ import annotations

# Re-export from DB layer so pipeline code has one import point
from Database.kg_constants import NODE_TYPES, RELATION_TYPES, _STOPWORDS  # noqa: F401


# ===========================================================================
# Pipeline tuning
# ===========================================================================

# Number of memories processed in a single sleep batch.
# Smaller = safer (fewer LLM calls fail together), slower overall.
# Larger = faster, but one bad LLM response corrupts more.
# 10 is a conservative default — tune up once the pipeline is proven stable.
BATCH_SIZE: int = 10

# Maximum number of sleep batches per /sleep invocation.
# Prevents a single /sleep command from running indefinitely on a large
# backlog. The user can type /sleep again to continue.
MAX_BATCHES_PER_SLEEP: int = 20

# Minimum confidence for an extracted entity to be considered for node
# creation. Below this, the entity is logged but not written.
MIN_ENTITY_CONFIDENCE: float = 0.5

# Minimum confidence for a proposed edge to be written.
# Higher than MIN_ENTITY_CONFIDENCE because edges are harder to validate —
# a node that turns out to be wrong is one row to fix, a wrong edge
# corrupts the graph's topology.
MIN_EDGE_CONFIDENCE: float = 0.6

# Confidence required before two candidate nodes are considered the same
# entity and one is renamed/merged into the other.
# Set deliberately high — auto-merge is the most destructive operation
# the sleep pipeline can perform. Below this, the match is noted in
# kg_graph_logs but no merge happens.
MERGE_CONFIDENCE_THRESHOLD: float = 0.85

# Number of hops to traverse when fetching the local subgraph for the
# operation proposer. 1 = only direct neighbours of the resolved nodes.
# 2 would give richer context but costs more tokens per proposal call.
SUBGRAPH_MAX_HOPS: int = 1

# Maximum number of nodes to include in the subgraph context sent to
# the operation proposer. If traversal finds more than this, the highest-
# importance nodes are kept and the rest are truncated.
SUBGRAPH_MAX_NODES: int = 15

# Maximum edges per node included in the subgraph context.
SUBGRAPH_MAX_EDGES_PER_NODE: int = 5

# LLM call timeouts (seconds)
ENTITY_EXTRACTION_TIMEOUT: float = 45.0
OPERATION_PROPOSAL_TIMEOUT: float = 45.0

# Max tokens for each LLM call
ENTITY_EXTRACTION_MAX_TOKENS: int = 800
OPERATION_PROPOSAL_MAX_TOKENS: int = 600

# Temperature for both LLM calls.
# Lower than the main model — this is a classification/extraction task,
# not a creative one. Slight non-zero temperature helps the model avoid
# getting stuck on repeated outputs across batches.
PIPELINE_TEMPERATURE: float = 0.2


# ===========================================================================
# Relation metadata
# ===========================================================================
# For each relation type: description for the LLM prompt, default weight
# used when inserting a new edge, and whether direction matters semantically.
#
# symmetric=True means A→B and B→A are equivalent (related_to, knows).
# These are stored as a single directed edge by convention: lower node
# name alphabetically is always the source. The validator enforces this.
#
# default_weight is the starting traversal weight assigned when an edge
# is first inserted. Updated by the sleep pipeline as confidence grows.

RELATION_META: dict[str, dict] = {
    "uses": {
        "description": "One entity actively uses another as a tool, service, or resource.",
        "example": "Seven uses PostgreSQL",
        "default_weight": 0.7,
        "symmetric": False,
    },
    "built_with": {
        "description": "One entity was constructed using another as a component or language.",
        "example": "Seven built_with Python",
        "default_weight": 0.8,
        "symmetric": False,
    },
    "depends_on": {
        "description": "One entity requires another to function correctly.",
        "example": "FastAPI depends_on Starlette",
        "default_weight": 0.75,
        "symmetric": False,
    },
    "contains": {
        "description": "One entity is a container or parent of another.",
        "example": "Seven's Backend contains PostgreSQL",
        "default_weight": 0.7,
        "symmetric": False,
    },
    "created": {
        "description": "One entity (usually a Person or Organization) created another.",
        "example": "Anthropic created Claude",
        "default_weight": 0.8,
        "symmetric": False,
    },
    "replaced": {
        "description": "One entity superseded or replaced another in a specific context.",
        "example": "PostgreSQL replaced SQLite",
        "default_weight": 0.75,
        "symmetric": False,
    },
    "part_of": {
        "description": "One entity is a component or member of a larger entity.",
        "example": "PostgreSQL part_of Seven's Backend",
        "default_weight": 0.7,
        "symmetric": False,
    },
    "related_to": {
        "description": "Two entities are connected but the exact relationship is unclear or general.",
        "example": "FastAPI related_to Python",
        "default_weight": 0.4,
        "symmetric": True,   # direction does not matter — always stored lower→higher name
    },
    "located_in": {
        "description": "One entity is physically or logically located within another.",
        "example": "Anthropic located_in San Francisco",
        "default_weight": 0.65,
        "symmetric": False,
    },
    "knows": {
        "description": "Two people or entities have a connection or awareness of each other.",
        "example": "Alice knows Bob",
        "default_weight": 0.5,
        "symmetric": True,
    },
    "worked_on": {
        "description": "A person or team contributed work to a project or entity.",
        "example": "Alice worked_on Seven",
        "default_weight": 0.7,
        "symmetric": False,
    },
    "prefers": {
        "description": "The user has a preference for one entity over alternatives.",
        "example": "User prefers Python",
        "default_weight": 0.6,
        "symmetric": False,
    },
    "learned": {
        "description": "A person or entity acquired knowledge or skill in another.",
        "example": "Alice learned FastAPI",
        "default_weight": 0.6,
        "symmetric": False,
    },
    "mentions": {
        "description": "One entity references or discusses another without a stronger relationship.",
        "example": "Session mentions PostgreSQL",
        "default_weight": 0.3,  # weakest relation — low traversal priority
        "symmetric": False,
    },
    "contradicts": {
        "description": "One piece of knowledge conflicts with another.",
        "example": "Memory_A contradicts Memory_B",
        "default_weight": 0.5,
        "symmetric": True,
    },
}

# Convenience: frozenset of symmetric relation types for fast membership check
SYMMETRIC_RELATIONS: frozenset[str] = frozenset(
    r for r, meta in RELATION_META.items() if meta["symmetric"]
)

# Convenience: map relation → default_weight for edge insertion
RELATION_DEFAULT_WEIGHTS: dict[str, float] = {
    r: meta["default_weight"] for r, meta in RELATION_META.items()
}


# ===========================================================================
# Node type guidance for LLM prompts
# ===========================================================================
# Plain-English descriptions included in both prompt templates so the LLM
# picks the most specific type rather than defaulting everything to Concept.

NODE_TYPE_GUIDANCE: dict[str, str] = {
    "Person":       "A specific individual — user, developer, researcher, or historical figure.",
    "Project":      "A software project, product, app, or system being built or used.",
    "Technology":   "A programming language, framework, library, tool, database, API, or protocol.",
    "Organization": "A company, team, open-source community, or institution.",
    "Place":        "A physical or logical location — city, country, server region, data center.",
    "Event":        "A discrete occurrence — conference, release, incident, milestone.",
    "Concept":      "An abstract idea, methodology, pattern, or topic that doesn't fit above.",
}


# ===========================================================================
# Prompt 1 — Entity extraction
# ===========================================================================
# Called by entity_extractor.py.
# Input: one or more memory texts (episodic or semantic).
# Output: entities and candidate relations to be resolved against the graph.
#
# DESIGN NOTES:
#   - The prompt explicitly gives the controlled type and relation vocabularies
#     so the model never invents its own. Local models have a strong tendency
#     to produce "is_a", "has", "type_of" etc. if not constrained.
#   - aliases are important — the model often sees "postgres", "pgsql", and
#     "PostgreSQL" in a single memory batch. Capturing these here prevents
#     three separate nodes from being created.
#   - The model is told to be conservative: only extract entities that are
#     clearly present, not entities it infers might exist.
#   - candidate_relations is extracted here as a hint only — the operation
#     proposer (Prompt 2) makes the final decision after seeing the subgraph.
#   - The model is told to output ONLY JSON. No preamble, no explanation.
#     Any non-JSON output from a local model is treated as a failed call.

ENTITY_EXTRACTION_SYSTEM: str = """\
You are Seven's Knowledge Graph entity extractor.

Your job is to read one or more memory texts and identify the important
concepts (entities) mentioned in them, along with candidate relationships
between those concepts.

ENTITY TYPES (use exactly one of these):
{node_type_guidance}

RELATION TYPES (use exactly one of these for candidate_relations):
{relation_list}

RULES:
- Only extract entities that are CLEARLY AND EXPLICITLY present in the text.
  Do not infer entities that are implied but not stated.
- For each entity, provide up to 3 aliases: alternative names or
  abbreviations you see in the text that refer to the same concept.
- heading must be 4-5 words maximum describing what the entity is.
- confidence is how certain you are this entity is meaningfully present
  (0.0-1.0). Be conservative: 0.5 = mentioned once, 0.8 = central topic,
  1.0 = explicitly the main subject.
- For candidate_relations: only propose a relation if BOTH entities appear
  in your entities list and the text clearly implies the relationship.
  Use the most specific relation type that fits.
- If nothing worth extracting is present, return an empty entities list.
- Respond ONLY with valid JSON. No markdown fences, no explanation.

OUTPUT FORMAT:
{{
  "entities": [
    {{
      "name": "canonical name in title case",
      "type": "one of the allowed types",
      "heading": "4-5 word description",
      "aliases": ["alias1", "alias2"],
      "confidence": 0.8
    }}
  ],
  "candidate_relations": [
    {{
      "source": "entity name exactly as it appears in entities list",
      "relation": "one of the allowed relation types",
      "target": "entity name exactly as it appears in entities list",
      "confidence": 0.7,
      "reasoning": "one sentence explaining why this relation exists in the text"
    }}
  ]
}}"""

# Filled at call time by entity_extractor.py using .format()
def build_entity_extraction_system() -> str:
    """
    Build the entity extraction system prompt with the controlled
    vocabularies injected. Called once per pipeline run, not per batch,
    since the vocabularies don't change.
    """
    node_type_guidance = "\n".join(
        f"  {name}: {desc}"
        for name, desc in NODE_TYPE_GUIDANCE.items()
    )
    relation_list = ", ".join(sorted(RELATION_TYPES))
    return ENTITY_EXTRACTION_SYSTEM.format(
        node_type_guidance=node_type_guidance,
        relation_list=relation_list,
    )


def build_entity_extraction_user(memory_texts: list[str]) -> str:
    """
    Build the user message for entity extraction.
    Each memory is numbered and separated clearly.
    """
    if not memory_texts:
        return "No memories provided."
    parts = [f"MEMORIES TO PROCESS ({len(memory_texts)} total):", ""]
    for i, text in enumerate(memory_texts, 1):
        parts.append(f"[Memory {i}]")
        parts.append(text.strip())
        parts.append("")
    return "\n".join(parts)


# ===========================================================================
# Prompt 2 — Operation proposal
# ===========================================================================
# Called by operation_proposer.py.
# Input: resolved entities (with their graph node ids) + local subgraph.
# Output: concrete graph operations to apply.
#
# DESIGN NOTES:
#   - At this point entity resolution has already happened — the proposer
#     receives node IDs, not names. This prevents the model from inventing
#     node IDs or proposing nodes that don't exist.
#   - The subgraph context shows what already exists so the proposer can
#     decide whether to insert a new edge, update confidence on an existing
#     one, or deactivate a superseded one.
#   - Four operation types: insert_edge, update_edge_confidence,
#     deactivate_edge, add_alias. No node creation here — that was
#     entity resolver's job.
#   - The model is told explicitly: "If no operations are needed, return
#     an empty operations list." This is the correct output when the graph
#     already accurately reflects the memories — not an error.
#   - evidence_memory_ids must be ids from the provided memory list.
#     The validator rejects any id not in that list.

OPERATION_PROPOSAL_SYSTEM: str = """\
You are Seven's Knowledge Graph operation proposer.

Resolved entities have already been matched to graph nodes. Your job is to
propose the MINIMUM set of graph operations that accurately captures the
relationships in the provided memories, given what already exists in the graph.

ALLOWED OPERATION TYPES:

1. insert_edge
   Create a new directed relationship between two nodes.
   Required fields: source_id, target_id, relation, confidence, evidence_memory_ids
   Use when: a relationship exists in the memories but NOT in the current subgraph.

2. update_edge_confidence
   Adjust the confidence of an EXISTING edge (provide its edge_id).
   Required fields: edge_id, confidence
   Use when: the memories provide additional evidence for or against an edge
   that already exists. Do NOT insert a duplicate — update the existing one.

3. deactivate_edge
   Mark an existing edge as no longer current (provide its edge_id).
   Required fields: edge_id, reasoning
   Use when: the memories indicate a relationship has changed or been superseded.
   Example: if the graph has "Seven uses SQLite" but the memory says Seven
   migrated to PostgreSQL, deactivate the SQLite edge.

4. add_alias
   Add an alternative name to an existing node.
   Required fields: node_id, alias
   Use when: the memories use a name for a concept that isn't already an alias
   for that node.

RELATION TYPES (for insert_edge only):
{relation_descriptions}

RULES:
- Use node ids and edge ids EXACTLY as provided in the resolved entities and
  current subgraph sections. Never invent ids.
- Only propose insert_edge if the relationship does NOT already exist as an
  active edge in the current subgraph. If it exists, use update_edge_confidence.
- confidence must be 0.0-1.0. Use the evidence in the memories, not assumption.
- evidence_memory_ids must only contain ids from the provided memory list.
- If the memories add nothing new to the graph, return an empty operations list.
  That is the correct output, not an error.
- Respond ONLY with valid JSON. No markdown, no explanation.

OUTPUT FORMAT:
{{
  "operations": [
    {{
      "type": "insert_edge",
      "source_id": "...",
      "target_id": "...",
      "relation": "...",
      "confidence": 0.8,
      "evidence_memory_ids": ["mem-id-1"]
    }},
    {{
      "type": "update_edge_confidence",
      "edge_id": "...",
      "confidence": 0.9
    }},
    {{
      "type": "deactivate_edge",
      "edge_id": "...",
      "reasoning": "..."
    }},
    {{
      "type": "add_alias",
      "node_id": "...",
      "alias": "..."
    }}
  ]
}}"""


def build_operation_proposal_system() -> str:
    """
    Build the operation proposal system prompt with the relation
    vocabulary injected. Called once per pipeline run.
    """
    relation_descriptions = "\n".join(
        f"  {rel}: {RELATION_META[rel]['description']}  (e.g. {RELATION_META[rel]['example']})"
        for rel in sorted(RELATION_TYPES)
    )
    return OPERATION_PROPOSAL_SYSTEM.format(
        relation_descriptions=relation_descriptions,
    )


def build_operation_proposal_user(
    resolved_entities: list[dict],
    subgraph_text: str,
    memory_ids_and_texts: list[tuple[str, str]],
) -> str:
    """
    Build the user message for the operation proposal call.

    resolved_entities: list of dicts with keys: name, node_id, type, heading,
        is_new (bool — whether this node was just created by entity_resolver).
    subgraph_text: formatted subgraph context from subgraph_retriever.
    memory_ids_and_texts: list of (memory_id, text) pairs for the current batch.
    """
    parts = []

    # Section 1: resolved entities with their graph node ids
    parts.append("RESOLVED ENTITIES (matched to graph nodes):")
    for ent in resolved_entities:
        status = "NEW NODE" if ent.get("is_new") else "existing node"
        parts.append(
            f"  name={ent['name']}  node_id={ent['node_id']}  "
            f"type={ent['type']}  ({status})"
        )
    parts.append("")

    # Section 2: current subgraph (what already exists)
    parts.append("CURRENT SUBGRAPH (existing nodes and edges near resolved entities):")
    parts.append(subgraph_text if subgraph_text.strip() else "  (no existing connections)")
    parts.append("")

    # Section 3: the memories being processed
    parts.append(f"MEMORIES ({len(memory_ids_and_texts)} total):")
    for mem_id, text in memory_ids_and_texts:
        parts.append(f"  [{mem_id}]: {text.strip()}")
    parts.append("")

    parts.append(
        "Propose the minimum graph operations that accurately reflect "
        "the relationships in these memories given the existing subgraph above."
    )
    return "\n".join(parts)


# ===========================================================================
# Validator rules (referenced by validator.py)
# ===========================================================================
# Centralised here so the validator imports these rather than hardcoding them.

# Operations the validator accepts. Any operation type not in this set is
# rejected immediately without looking at its fields.
VALID_OPERATION_TYPES: frozenset[str] = frozenset({
    "insert_edge",
    "update_edge_confidence",
    "deactivate_edge",
    "add_alias",
})

# Required fields for each operation type.
# The validator checks that all required fields are present and non-empty
# before passing the operation to the DB layer.
OPERATION_REQUIRED_FIELDS: dict[str, list[str]] = {
    "insert_edge":            ["source_id", "target_id", "relation", "confidence"],
    "update_edge_confidence": ["edge_id", "confidence"],
    "deactivate_edge":        ["edge_id"],
    "add_alias":              ["node_id", "alias"],
}

# Confidence bounds — the validator clamps to these rather than rejecting.
# A value of 1.02 from the LLM is almost certainly a rounding artefact,
# not a sign of a fundamentally bad operation.
CONFIDENCE_MIN: float = 0.0
CONFIDENCE_MAX: float = 1.0

def build_entity_extraction_user_from_bundle(bundle) -> str:
    parts = [f"SESSION: {bundle.session_id}", ""]
    if bundle.conversation_text and bundle.conversation_text.strip():
        parts.append("CONVERSATION NARRATIVE (chunk summaries — primary extraction signal):")
        parts.append(bundle.conversation_text.strip())
    else:
        parts.append("CONVERSATION NARRATIVE: (not available for this session)")
    parts.append("")
    if bundle.episode_text and bundle.episode_text.strip():
        parts.append("EPISODE SUMMARY (title and overview):")
        parts.append(bundle.episode_text.strip())
        parts.append("")
    if bundle.semantic_texts:
        parts.append(f"SEMANTIC FACTS FROM THIS SESSION ({len(bundle.semantic_texts)}):")
        for i, text in enumerate(bundle.semantic_texts, 1):
            parts.append(f"  {i}. {text.strip()}")
        parts.append("")
    parts.append(
        "Extract all entities and relationships present in this session. "
        "Use the CONVERSATION NARRATIVE as the primary signal. "
        "Semantic facts confirm and reinforce — do not treat them as the only source."
    )
    return "\n".join(parts)