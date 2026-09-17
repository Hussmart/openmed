from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import pytest

from openmed.graph.builder import EntityGraph, build_entity_graph


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


def _entity_analyzer(fixtures: Dict[str, Dict[str, List[_FakeEntity]]]):
    """Return an analyzer(text, model_name) reading canned entities by text+model."""

    def analyzer(text: str, model_name: str) -> _FakeResult:
        return _FakeResult(entities=list(fixtures.get(text, {}).get(model_name, [])))

    return analyzer


class TestBuildEntityGraphValidation:
    def test_rejects_empty_model_names(self) -> None:
        with pytest.raises(ValueError, match="model_names"):
            build_entity_graph(["hello"], model_names=())

    def test_rejects_invalid_cooccurrence_window(self) -> None:
        with pytest.raises(ValueError, match="cooccurrence_window"):
            build_entity_graph(["hello"], cooccurrence_window="paragraph")  # type: ignore[arg-type]


class TestBuildEntityGraphSentenceWindow:
    def test_links_entities_in_the_same_sentence(self) -> None:
        text = "Patient takes Metformin for Diabetes."
        fixtures = {
            text: {
                "disease_detection_superclinical": [
                    _FakeEntity("Metformin", "Drug", start=14, end=23),
                    _FakeEntity("Diabetes", "Disease", start=28, end=36),
                ]
            }
        }
        graph = build_entity_graph(
            [("note-1", text)],
            model_names=("disease_detection_superclinical",),
            analyzer=_entity_analyzer(fixtures),
        )

        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1
        edge = next(iter(graph.edges.values()))
        assert edge.weight == 1
        assert edge.relation == "co_occurs_with"

    def test_does_not_link_entities_in_different_sentences(self) -> None:
        text = "Patient takes Metformin daily. Family history of Asthma noted."
        fixtures = {
            text: {
                "disease_detection_superclinical": [
                    _FakeEntity("Metformin", "Drug", start=14, end=23),
                    _FakeEntity("Asthma", "Disease", start=51, end=57),
                ]
            }
        }
        graph = build_entity_graph(
            [("note-1", text)],
            cooccurrence_window="sentence",
            analyzer=_entity_analyzer(fixtures),
        )

        assert len(graph.nodes) == 2
        assert len(graph.edges) == 0

    def test_document_window_links_entities_across_sentences(self) -> None:
        text = "Patient takes Metformin daily. Family history of Asthma noted."
        fixtures = {
            text: {
                "disease_detection_superclinical": [
                    _FakeEntity("Metformin", "Drug", start=14, end=23),
                    _FakeEntity("Asthma", "Disease", start=51, end=57),
                ]
            }
        }
        graph = build_entity_graph(
            [("note-1", text)],
            cooccurrence_window="document",
            analyzer=_entity_analyzer(fixtures),
        )

        assert len(graph.edges) == 1


class TestBuildEntityGraphAggregation:
    def test_edge_weight_accumulates_across_documents(self) -> None:
        text_a = "Metformin helps Diabetes."
        text_b = "Doctor prescribed Metformin for Diabetes again."
        fixtures = {
            text_a: {
                "m": [
                    _FakeEntity("Metformin", "Drug", start=0, end=9),
                    _FakeEntity("Diabetes", "Disease", start=16, end=24),
                ]
            },
            text_b: {
                "m": [
                    _FakeEntity("Metformin", "Drug", start=19, end=28),
                    _FakeEntity("Diabetes", "Disease", start=33, end=41),
                ]
            },
        }
        graph = build_entity_graph(
            {"doc-a": text_a, "doc-b": text_b},
            model_names=("m",),
            analyzer=_entity_analyzer(fixtures),
        )

        edge = next(iter(graph.edges.values()))
        assert edge.weight == 2
        assert len(edge.document_ids) == 2

    def test_confidence_threshold_drops_low_confidence_entities(self) -> None:
        text = "Metformin helps Diabetes."
        fixtures = {
            text: {
                "m": [
                    _FakeEntity("Metformin", "Drug", start=0, end=9, confidence=0.2),
                    _FakeEntity(
                        "Diabetes", "Disease", start=16, end=24, confidence=0.95
                    ),
                ]
            }
        }
        graph = build_entity_graph(
            [text],
            model_names=("m",),
            confidence_threshold=0.5,
            analyzer=_entity_analyzer(fixtures),
        )

        assert len(graph.nodes) == 1
        assert len(graph.edges) == 0

    def test_multiple_models_contribute_to_the_same_graph(self) -> None:
        text = "Metformin helps Diabetes via AMPK activation."
        fixtures = {
            text: {
                "disease_model": [
                    _FakeEntity("Metformin", "Drug", start=0, end=9),
                    _FakeEntity("Diabetes", "Disease", start=16, end=24),
                ],
                "gene_model": [
                    _FakeEntity("AMPK", "Gene", start=30, end=34),
                ],
            }
        }
        graph = build_entity_graph(
            [text],
            model_names=("disease_model", "gene_model"),
            cooccurrence_window="document",
            analyzer=_entity_analyzer(fixtures),
        )

        assert len(graph.nodes) == 3
        assert len(graph.edges) == 3  # every pair among 3 entities


class TestEntityGraphQueries:
    def _sample_graph(self) -> EntityGraph:
        text = "Metformin helps Diabetes. Aspirin helps Fever."
        fixtures = {
            text: {
                "m": [
                    _FakeEntity("Metformin", "Drug", start=0, end=9),
                    _FakeEntity("Diabetes", "Disease", start=16, end=24),
                    _FakeEntity("Aspirin", "Drug", start=26, end=33),
                    _FakeEntity("Fever", "Disease", start=41, end=46),
                ]
            }
        }
        return build_entity_graph(
            [text], model_names=("m",), analyzer=_entity_analyzer(fixtures)
        )

    def test_neighbors_returns_only_connected_nodes(self) -> None:
        graph = self._sample_graph()
        metformin_key = "Drug:metformin"

        neighbors = graph.neighbors(metformin_key)

        neighbor_keys = {node.key for node, _edge in neighbors}
        assert neighbor_keys == {"Disease:diabetes"}

    def test_neighbors_respects_top_k(self) -> None:
        graph = self._sample_graph()
        assert graph.neighbors("Drug:metformin", top_k=0) == []

    def test_find_node_is_case_insensitive(self) -> None:
        graph = self._sample_graph()

        node = graph.find_node("METFORMIN")

        assert node is not None
        assert node.label == "Drug"

    def test_find_node_returns_none_for_unknown_entity(self) -> None:
        graph = self._sample_graph()

        assert graph.find_node("Ibuprofen") is None

    def test_to_dict_from_dict_round_trip_preserves_counts(self) -> None:
        graph = self._sample_graph()

        payload = graph.to_dict()
        rebuilt = EntityGraph.from_dict(payload)

        assert len(rebuilt.nodes) == len(graph.nodes)
        assert len(rebuilt.edges) == len(graph.edges)
        # Raw document ids are deliberately not serialized.
        assert all("document_ids" not in node for node in payload["nodes"])
        assert all("document_ids" not in edge for edge in payload["edges"])
