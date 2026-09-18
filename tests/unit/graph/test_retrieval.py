from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pytest

from openmed.graph.builder import build_entity_graph
from openmed.graph.retrieval import GraphRAGRetriever


@dataclass
class _FakeEntity:
    text: str
    label: str
    start: int
    end: int
    confidence: float = 0.9


@dataclass
class _FakeResult:
    entities: List[_FakeEntity]


def _corpus_graph():
    text = "Metformin helps Diabetes. Aspirin helps Fever."
    fixtures = {
        text: [
            _FakeEntity("Metformin", "Drug", start=0, end=9),
            _FakeEntity("Diabetes", "Disease", start=16, end=24),
            _FakeEntity("Aspirin", "Drug", start=26, end=33),
            _FakeEntity("Fever", "Disease", start=41, end=46),
        ]
    }

    def analyzer(analyzed_text: str, model_name: str) -> _FakeResult:
        return _FakeResult(entities=list(fixtures.get(analyzed_text, [])))

    return build_entity_graph([text], model_names=("m",), analyzer=analyzer)


def _query_analyzer(entities_by_query: dict):
    def analyzer(text: str, model_name: str) -> _FakeResult:
        return _FakeResult(entities=list(entities_by_query.get(text, [])))

    return analyzer


class TestGraphRAGRetrieverValidation:
    def test_rejects_non_positive_top_k(self) -> None:
        with pytest.raises(ValueError, match="top_k_per_entity"):
            GraphRAGRetriever(_corpus_graph(), top_k_per_entity=0)


class TestGraphRAGRetrieverRetrieve:
    def test_returns_neighbors_of_a_recognized_query_entity(self) -> None:
        graph = _corpus_graph()
        query = "Does Metformin interact with anything?"
        analyzer = _query_analyzer(
            {query: [_FakeEntity("Metformin", "Drug", start=5, end=14)]}
        )
        retriever = GraphRAGRetriever(graph, model_names=("m",), analyzer=analyzer)

        context = retriever.retrieve(query)

        assert context.matched_entities == ("Metformin",)
        assert len(context.triples) == 1
        source, relation, target, weight = context.triples[0]
        assert {source, target} == {"Metformin", "Diabetes"}
        assert relation == "co_occurs_with"
        assert "Metformin" in context.context_text
        assert "Diabetes" in context.context_text

    def test_query_with_no_recognized_entities_produces_empty_context(self) -> None:
        graph = _corpus_graph()
        query = "How is the weather today?"
        analyzer = _query_analyzer({query: []})
        retriever = GraphRAGRetriever(graph, model_names=("m",), analyzer=analyzer)

        context = retriever.retrieve(query)

        assert context.matched_entities == ()
        assert context.triples == ()
        assert "No known entities" in context.context_text

    def test_query_entity_unknown_to_graph_yields_no_triples(self) -> None:
        graph = _corpus_graph()
        query = "Tell me about Ibuprofen."
        analyzer = _query_analyzer(
            {query: [_FakeEntity("Ibuprofen", "Drug", start=14, end=23)]}
        )
        retriever = GraphRAGRetriever(graph, model_names=("m",), analyzer=analyzer)

        context = retriever.retrieve(query)

        assert context.matched_entities == ()
        assert context.triples == ()

    def test_to_dict_matches_context_fields(self) -> None:
        graph = _corpus_graph()
        query = "Does Metformin interact with anything?"
        analyzer = _query_analyzer(
            {query: [_FakeEntity("Metformin", "Drug", start=5, end=14)]}
        )
        retriever = GraphRAGRetriever(graph, model_names=("m",), analyzer=analyzer)

        payload = retriever.retrieve(query).to_dict()

        assert payload["query"] == query
        assert payload["matched_entities"] == ["Metformin"]
        assert payload["triples"][0]["relation"] == "co_occurs_with"
