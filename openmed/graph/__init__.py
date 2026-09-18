"""Local biomedical entity graph construction and graph-grounded retrieval.

Turns OpenMed's per-document NER output into a queryable co-occurrence graph
(:mod:`openmed.graph.builder`) and uses it to ground free-text questions in
relationships actually observed in a corpus, for use as LLM context
(:mod:`openmed.graph.retrieval`). See :mod:`openmed.graph.builder` for how
this differs from -- and composes with -- :mod:`openmed.interop.retrieval`.
"""

from __future__ import annotations

from .builder import EntityGraph, GraphEdge, GraphNode, build_entity_graph
from .retrieval import GraphContext, GraphRAGRetriever

__all__ = [
    "EntityGraph",
    "GraphContext",
    "GraphEdge",
    "GraphNode",
    "GraphRAGRetriever",
    "build_entity_graph",
]
