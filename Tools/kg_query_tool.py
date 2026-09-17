"""
Tools/kg_query_tool.py

Thin LLM-facing bridge to KnowledgeGraph.kg_query_service — mirrors the
shape of Tools/semantic_memory_tool.py. The LLM never touches the graph
DB or subgraph_retriever directly.
"""

import KnowledgeGraph.kg_query_service as kg_query_service
import Tools.scratchpad_tool as scratchpad_tool
from GlobalHelpers.logger import get_logger

log = get_logger(__name__)


def query_knowledge_graph_tool(query: str) -> str:
    """
    Look up entities and their relationships in the Knowledge Graph for
    `query`. Returns formatted context text (seed nodes + active edges),
    or a fixed no-match message if nothing relevant was found.
    """
    try:
        text = kg_query_service.query_knowledge_graph(query)
    except Exception as e:
        log.error("query_knowledge_graph_tool: error for query=%r: %s", query, e, exc_info=True)
        text = "Knowledge graph lookup failed."

    scratchpad_tool.add_scratchpad_tool_output("query_knowledge_graph", text)
    return text