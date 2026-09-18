"""Ground LLM answers in an :class:`~openmed.graph.builder.EntityGraph`.

:class:`GraphRAGRetriever` turns a free-text question into a small set of
entity-relationship triples the graph actually observed, formatted as plain
text a prompt can include directly. It deliberately returns framework-neutral
data (plain dataclasses, not a LangChain ``Document`` or a LlamaIndex node) so
it composes with any RAG framework -- including the LangChain/LlamaIndex
adapters already in :mod:`openmed.interop` -- without adding a hard dependency
on one of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .builder import EntityGraph, GraphEdge, GraphNode

__all__ = ["GraphContext", "GraphRAGRetriever"]

_DEFAULT_MODEL_NAMES: Tuple[str, ...] = ("disease_detection_superclinical",)


@dataclass(frozen=True)
class GraphContext:
    """Graph-grounded context for one query, ready to inject into a prompt."""

    query: str
    matched_entities: Tuple[str, ...]
    triples: Tuple[
        Tuple[str, str, str, int], ...
    ]  # (entity_a, relation, entity_b, weight)
    context_text: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "matched_entities": list(self.matched_entities),
            "triples": [
                {"source": a, "relation": relation, "target": b, "weight": weight}
                for a, relation, b, weight in self.triples
            ],
            "context_text": self.context_text,
        }


class GraphRAGRetriever:
    """Extract query entities and return their graph neighborhood as context.

    Example:
        >>> retriever = GraphRAGRetriever(graph)  # doctest: +SKIP
        >>> context = retriever.retrieve("Does metformin interact with the patient's condition?")  # doctest: +SKIP
        >>> print(context.context_text)  # doctest: +SKIP
    """

    def __init__(
        self,
        graph: EntityGraph,
        *,
        model_names: Sequence[str] = _DEFAULT_MODEL_NAMES,
        top_k_per_entity: int = 5,
        confidence_threshold: float = 0.5,
        analyzer: Optional[Callable[[str, str], Any]] = None,
    ) -> None:
        if top_k_per_entity <= 0:
            raise ValueError("top_k_per_entity must be positive")
        self._graph = graph
        self._model_names = tuple(model_names)
        self._top_k_per_entity = top_k_per_entity
        self._confidence_threshold = confidence_threshold
        self._analyzer = analyzer or _default_analyzer

    def retrieve(self, query: str) -> GraphContext:
        """Return the graph neighborhood of every entity mentioned in ``query``."""

        query_entities = self._extract_query_entities(query)
        matched_nodes: List[GraphNode] = []
        for label, surface in query_entities:
            node = self._graph.find_node(surface, label=label) or self._graph.find_node(
                surface
            )
            if node is not None:
                matched_nodes.append(node)

        triples: List[Tuple[str, str, str, int]] = []
        seen_pairs: set = set()
        for node in matched_nodes:
            for neighbor, edge in self._graph.neighbors(
                node.key, top_k=self._top_k_per_entity
            ):
                pair = tuple(sorted((node.key, neighbor.key)))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                triples.append(
                    (
                        _display_name(node),
                        edge.relation,
                        _display_name(neighbor),
                        edge.weight,
                    )
                )

        triples.sort(key=lambda triple: triple[3], reverse=True)
        return GraphContext(
            query=query,
            matched_entities=tuple(_display_name(node) for node in matched_nodes),
            triples=tuple(triples),
            context_text=_render_context_text(query, matched_nodes, triples),
        )

    def _extract_query_entities(self, query: str) -> List[Tuple[str, str]]:
        found: List[Tuple[str, str]] = []
        for model_name in self._model_names:
            result = self._analyzer(query, model_name)
            for entity in getattr(result, "entities", None) or ():
                confidence = getattr(entity, "confidence", None)
                if confidence is not None and confidence < self._confidence_threshold:
                    continue
                text = getattr(entity, "text", "") or ""
                if text.strip():
                    found.append((getattr(entity, "label", "") or "ENTITY", text))
        return found


def _default_analyzer(text: str, model_name: str) -> Any:
    from openmed import analyze_text

    return analyze_text(text, model_name=model_name)


def _display_name(node: GraphNode) -> str:
    forms = node.surface_forms
    return forms[0] if forms else node.key.split(":", 1)[-1]


def _render_context_text(
    query: str,
    matched_nodes: Sequence[GraphNode],
    triples: Sequence[Tuple[str, str, str, int]],
) -> str:
    if not matched_nodes:
        return f"No known entities from the local graph were found in: {query!r}"

    lines = [
        f"Entities recognized in the query: {', '.join(_display_name(n) for n in matched_nodes)}.",
        "Relationships observed in the local corpus (not verified medical facts -- "
        "co-occurrence only):",
    ]
    if not triples:
        lines.append("(none found)")
    for source, relation, target, weight in triples:
        lines.append(
            f"- {source} {relation.replace('_', ' ')} {target} (seen {weight}x)"
        )
    return "\n".join(lines)
