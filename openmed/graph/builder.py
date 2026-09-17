"""Build a local entity co-occurrence graph from OpenMed NER output.

OpenMed's NER models extract clinical entities (diseases, drugs, genes, ...)
one document at a time, but nothing links those entities *to each other*.
:func:`build_entity_graph` closes that gap: it runs one or more NER models
across a corpus and records which entities co-occur, within the same sentence
or the same document, producing a weighted graph that
:class:`openmed.graph.retrieval.GraphRAGRetriever` can then use to ground an
LLM's answer in relationships OpenMed actually observed -- instead of the
model inventing a drug-disease association that was never in the text.

This is a different problem from :mod:`openmed.interop.retrieval`, which
solves *safe* retrieval (keeping PHI out of an external LLM call). This module
solves *grounded* retrieval (keeping an LLM's answer tied to real extracted
relationships). The two compose: run a :class:`GraphRAGRetriever` over text
that has already been de-identified by the existing pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from openmed.core.config import OpenMedConfig
from openmed.processing.sentences import segment_clinical_text

__all__ = [
    "CooccurrenceWindow",
    "EntityGraph",
    "GraphEdge",
    "GraphNode",
    "build_entity_graph",
]

CooccurrenceWindow = str  # "sentence" | "document"

_DEFAULT_MODEL_NAMES: Tuple[str, ...] = ("disease_detection_superclinical",)


@dataclass(frozen=True)
class GraphNode:
    """One distinct entity, aggregated across every document it appeared in."""

    key: str
    label: str
    surface_forms: Tuple[str, ...]
    document_ids: Tuple[str, ...]
    mention_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "surface_forms": list(self.surface_forms),
            "document_count": len(self.document_ids),
            "mention_count": self.mention_count,
        }


@dataclass(frozen=True)
class GraphEdge:
    """A co-occurrence relationship between two entity nodes.

    ``relation`` is always ``"co_occurs_with"`` today -- this graph records
    observed co-occurrence, not a typed clinical relation (e.g. "treats" or
    "causes"). Callers that need typed relations should treat an edge as a
    candidate to verify, not an assertion.
    """

    source: str
    target: str
    weight: int
    document_ids: Tuple[str, ...]
    relation: str = "co_occurs_with"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "weight": self.weight,
            "document_count": len(self.document_ids),
        }


@dataclass
class EntityGraph:
    """A queryable, JSON-serializable entity co-occurrence graph."""

    nodes: Dict[str, GraphNode] = field(default_factory=dict)
    edges: Dict[Tuple[str, str], GraphEdge] = field(default_factory=dict)

    def find_node(
        self, text: str, *, label: Optional[str] = None
    ) -> Optional[GraphNode]:
        """Look up a node by surface text (case-insensitive), optionally by label."""

        key = _normalize_surface(text)
        for node in self.nodes.values():
            if label is not None and node.label != label:
                continue
            if key == _normalize_surface(node.key.split(":", 1)[-1]):
                return node
            if any(key == _normalize_surface(form) for form in node.surface_forms):
                return node
        return None

    def neighbors(
        self, node_key: str, *, top_k: int = 10
    ) -> List[Tuple[GraphNode, GraphEdge]]:
        """Return up to ``top_k`` neighboring nodes, ranked by co-occurrence weight."""

        matches: List[Tuple[GraphNode, GraphEdge]] = []
        for (source, target), edge in self.edges.items():
            if source == node_key:
                other_key = target
            elif target == node_key:
                other_key = source
            else:
                continue
            other = self.nodes.get(other_key)
            if other is not None:
                matches.append((other, edge))
        matches.sort(key=lambda pair: pair[1].weight, reverse=True)
        return matches[:top_k]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the graph, deliberately dropping raw document ids.

        ``GraphNode.to_dict``/``GraphEdge.to_dict`` emit only aggregate
        ``document_count`` values, not the ids themselves -- a shared
        ``graph.json`` should not let a reader infer which specific records
        contributed to an entity or relationship, mirroring the value-free
        aggregate pattern used by :mod:`openmed.risk.redaction_diff`.
        """

        return {
            "schema_version": 1,
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges.values()],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EntityGraph":
        """Rebuild a graph from :meth:`to_dict`'s output.

        Reconstructed ``document_ids`` are synthetic placeholders (``doc-0``,
        ``doc-1``, ...) that only preserve the original document *count* for
        :meth:`GraphNode.to_dict`/:meth:`GraphEdge.to_dict` round-tripping --
        the real ids were never serialized (see :meth:`to_dict`).
        """

        nodes = {
            row["key"]: GraphNode(
                key=row["key"],
                label=row["label"],
                surface_forms=tuple(row.get("surface_forms", ())),
                document_ids=tuple(
                    f"doc-{i}" for i in range(row.get("document_count", 0))
                ),
                mention_count=row.get("mention_count", 0),
            )
            for row in data.get("nodes", ())
        }
        edges = {
            (row["source"], row["target"]): GraphEdge(
                source=row["source"],
                target=row["target"],
                weight=row["weight"],
                document_ids=tuple(
                    f"doc-{i}" for i in range(row.get("document_count", 0))
                ),
                relation=row.get("relation", "co_occurs_with"),
            )
            for row in data.get("edges", ())
        }
        return cls(nodes=nodes, edges=edges)


def build_entity_graph(
    documents: Sequence[str] | Sequence[Tuple[str, str]] | Mapping[str, str],
    *,
    model_names: Sequence[str] = _DEFAULT_MODEL_NAMES,
    cooccurrence_window: CooccurrenceWindow = "sentence",
    confidence_threshold: float = 0.5,
    config: Optional[OpenMedConfig] = None,
    analyzer: Optional[Callable[[str, str], Any]] = None,
) -> EntityGraph:
    """Run NER across a corpus and build a weighted entity co-occurrence graph.

    Args:
        documents: Either plain strings (auto-numbered ``doc-0``, ``doc-1``,
            ...), ``(doc_id, text)`` pairs, or a ``{doc_id: text}`` mapping.
        model_names: OpenMed NER model keys to run over every document. Using
            several (e.g. a disease model and a drug model) lets the graph
            connect entity types a single model would never emit together.
        cooccurrence_window: ``"sentence"`` links only entities mentioned in
            the same sentence (fewer, more precise edges); ``"document"``
            links every entity pair in a document (more edges, coarser
            evidence).
        confidence_threshold: Entities below this confidence are dropped
            before graph construction.
        config: Optional :class:`~openmed.core.config.OpenMedConfig`.
        analyzer: Injectable ``(text, model_name) -> AnalyzeResult``, mainly
            for tests. Defaults to :func:`openmed.analyze_text`.

    Returns:
        The built :class:`EntityGraph`.
    """

    if cooccurrence_window not in ("sentence", "document"):
        raise ValueError('cooccurrence_window must be "sentence" or "document"')
    if not model_names:
        raise ValueError("model_names must not be empty")

    run_analysis = analyzer or _default_analyzer

    node_accum: Dict[str, Dict[str, Any]] = {}
    edge_accum: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for doc_id, text in _iter_documents(documents):
        entities: List[Tuple[str, str, str]] = []  # (node_key, label, surface_text)
        for model_name in model_names:
            result = run_analysis(text, model_name)
            for entity in _result_entities(result):
                confidence = getattr(entity, "confidence", None)
                if confidence is not None and confidence < confidence_threshold:
                    continue
                surface = getattr(entity, "text", "") or ""
                label = getattr(entity, "label", "") or "ENTITY"
                start = getattr(entity, "start", None)
                if not surface.strip():
                    continue
                node_key = _node_key(label, surface)
                _accumulate_node(node_accum, node_key, label, surface, doc_id)
                entities.append((node_key, start if start is not None else -1, surface))

        if cooccurrence_window == "sentence":
            buckets = _bucket_by_sentence(text, entities)
        else:
            buckets = [tuple(key for key, _start, _surface in entities)]

        for bucket in buckets:
            unique_keys = sorted(set(bucket))
            for i in range(len(unique_keys)):
                for j in range(i + 1, len(unique_keys)):
                    _accumulate_edge(edge_accum, unique_keys[i], unique_keys[j], doc_id)

    nodes = {
        key: GraphNode(
            key=key,
            label=data["label"],
            surface_forms=tuple(sorted(data["surface_forms"])),
            document_ids=tuple(sorted(data["document_ids"])),
            mention_count=data["mention_count"],
        )
        for key, data in node_accum.items()
    }
    edges = {
        pair: GraphEdge(
            source=pair[0],
            target=pair[1],
            weight=data["weight"],
            document_ids=tuple(sorted(data["document_ids"])),
        )
        for pair, data in edge_accum.items()
    }
    return EntityGraph(nodes=nodes, edges=edges)


def _default_analyzer(text: str, model_name: str) -> Any:
    from openmed import analyze_text

    return analyze_text(text, model_name=model_name)


def _result_entities(result: Any) -> Iterable[Any]:
    entities = getattr(result, "entities", None)
    if entities is not None:
        return entities
    if isinstance(result, Mapping):
        return result.get("entities", ())
    return ()


def _iter_documents(
    documents: Sequence[str] | Sequence[Tuple[str, str]] | Mapping[str, str],
) -> Iterable[Tuple[str, str]]:
    if isinstance(documents, Mapping):
        for doc_id, text in documents.items():
            yield str(doc_id), text
        return
    for index, item in enumerate(documents):
        if isinstance(item, tuple) and len(item) == 2:
            yield str(item[0]), item[1]
        else:
            yield f"doc-{index}", item  # type: ignore[arg-type]


def _node_key(label: str, surface: str) -> str:
    return f"{label}:{_normalize_surface(surface)}"


def _normalize_surface(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _accumulate_node(
    accum: Dict[str, Dict[str, Any]],
    node_key: str,
    label: str,
    surface: str,
    doc_id: str,
) -> None:
    entry = accum.setdefault(
        node_key,
        {
            "label": label,
            "surface_forms": set(),
            "document_ids": set(),
            "mention_count": 0,
        },
    )
    entry["surface_forms"].add(surface)
    entry["document_ids"].add(doc_id)
    entry["mention_count"] += 1


def _accumulate_edge(
    accum: Dict[Tuple[str, str], Dict[str, Any]],
    key_a: str,
    key_b: str,
    doc_id: str,
) -> None:
    if key_a == key_b:
        return
    pair = (key_a, key_b) if key_a < key_b else (key_b, key_a)
    entry = accum.setdefault(pair, {"weight": 0, "document_ids": set()})
    entry["weight"] += 1
    entry["document_ids"].add(doc_id)


def _bucket_by_sentence(
    text: str, entities: List[Tuple[str, int, str]]
) -> List[Tuple[str, ...]]:
    try:
        spans = segment_clinical_text(text)
    except Exception:
        return [tuple(key for key, _start, _surface in entities)]

    buckets: List[List[str]] = [[] for _ in spans]
    fallback: List[str] = []
    for node_key, start, _surface in entities:
        placed = False
        if start is not None and start >= 0:
            for index, span in enumerate(spans):
                if span.start <= start < span.end:
                    buckets[index].append(node_key)
                    placed = True
                    break
        if not placed:
            fallback.append(node_key)

    result = [tuple(bucket) for bucket in buckets if bucket]
    if fallback:
        result.append(tuple(fallback))
    return result
