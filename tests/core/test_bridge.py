"""
Unit tests for the bone-flesh bridge operations.

Core claims:
    - clause_from_edge lifts an edge to a clause, discarding confidence
    - edge_from_clause recovers an edge from a unit positive clause
    - edge_from_clause returns None for multi-literal or negative clauses
    - stiffen_edges raises matched edges to confidence 1.0
    - stiffen_to_limit raises matched edges to the limit, not beyond
    - Unmatched edges pass through both stiffen operations unchanged
"""

import pytest
from hypothesis import given, assume
from hypothesis import strategies as st

from otter.core.state import Edge, Clause
from otter.core.bridge import (
    clause_from_edge, edge_from_clause,
    stiffen_edges, stiffen_to_limit,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _edge(subj, pred, obj, conf=0.5):
    return Edge(subject=subj, predicate=pred, object=obj, confidence=conf)


def _unit_clause(pred, subj, obj, label="test"):
    return Clause(
        literals=frozenset({(True, pred, subj, obj)}),
        label=label,
    )


# ── clause_from_edge ─────────────────────────────────────────────────────────

class TestClauseFromEdge:
    def test_produces_correct_literal(self):
        edge = _edge("alice", "knows", "bob", 0.8)
        clause = clause_from_edge(edge)
        assert (True, "knows", "alice", "bob") in clause.literals

    def test_is_unit_clause(self):
        edge = _edge("a", "p", "b")
        clause = clause_from_edge(edge)
        assert len(clause.literals) == 1

    def test_label_references_edge(self):
        edge = _edge("alice", "knows", "bob")
        clause = clause_from_edge(edge)
        assert "edge" in clause.label.lower()

    def test_confidence_not_in_clause(self):
        """The bone carries no confidence -- that's the point."""
        edge = _edge("a", "p", "b", 0.3)
        clause = clause_from_edge(edge)
        # Clause has no confidence attribute
        assert not hasattr(clause, "confidence")


# ── edge_from_clause ─────────────────────────────────────────────────────────

class TestEdgeFromClause:
    def test_recovers_edge_from_unit_positive_clause(self):
        clause = _unit_clause("knows", "alice", "bob")
        edge = edge_from_clause(clause)
        assert edge is not None
        assert edge.predicate == "knows"
        assert edge.subject == "alice"
        assert edge.object == "bob"

    def test_default_confidence_is_one(self):
        clause = _unit_clause("p", "a", "b")
        edge = edge_from_clause(clause)
        assert edge.confidence == 1.0

    def test_custom_confidence(self):
        clause = _unit_clause("p", "a", "b")
        edge = edge_from_clause(clause, confidence=0.7)
        assert edge.confidence == 0.7

    def test_returns_none_for_multi_literal_clause(self):
        clause = Clause(literals=frozenset({
            (True, "p", "a", "b"),
            (True, "q", "c", "d"),
        }))
        assert edge_from_clause(clause) is None

    def test_returns_none_for_negative_clause(self):
        clause = Clause(literals=frozenset({(False, "p", "a", "b")}))
        assert edge_from_clause(clause) is None

    def test_returns_none_for_empty_clause(self):
        clause = Clause(literals=frozenset())
        assert edge_from_clause(clause) is None


# ── Round-trip: edge → clause → edge ─────────────────────────────────────────

class TestBridgeRoundTrip:
    def test_round_trip_preserves_content(self):
        original = _edge("alice", "knows", "bob", 0.8)
        clause = clause_from_edge(original)
        recovered = edge_from_clause(clause)
        assert recovered is not None
        assert recovered.predicate == original.predicate
        assert recovered.subject == original.subject
        assert recovered.object == original.object

    def test_round_trip_sets_confidence_to_one(self):
        """The bridge lifts to bone (no confidence), then recovers at 1.0."""
        original = _edge("a", "p", "b", 0.3)
        clause = clause_from_edge(original)
        recovered = edge_from_clause(clause)
        assert recovered.confidence == 1.0  # proven → rigid


# ── stiffen_edges ────────────────────────────────────────────────────────────

class TestStiffenEdges:
    def test_matched_edge_stiffened_to_one(self):
        edges = [_edge("alice", "knows", "bob", 0.6)]
        proven = [_unit_clause("knows", "alice", "bob")]
        result = stiffen_edges(edges, proven)
        assert result[0].confidence == 1.0

    def test_unmatched_edge_unchanged(self):
        edges = [_edge("alice", "knows", "bob", 0.6)]
        proven = [_unit_clause("trusts", "carol", "dave")]
        result = stiffen_edges(edges, proven)
        assert result[0].confidence == 0.6

    def test_preserves_edge_identity(self):
        edges = [_edge("a", "p", "b", 0.5)]
        proven = [_unit_clause("p", "a", "b")]
        result = stiffen_edges(edges, proven)
        assert result[0].subject == "a"
        assert result[0].predicate == "p"
        assert result[0].object == "b"

    def test_multiple_edges_partial_stiffen(self):
        edges = [
            _edge("a", "p", "b", 0.3),
            _edge("c", "q", "d", 0.4),
            _edge("e", "r", "f", 0.5),
        ]
        proven = [_unit_clause("q", "c", "d")]
        result = stiffen_edges(edges, proven)
        assert result[0].confidence == 0.3  # unmatched
        assert result[1].confidence == 1.0  # matched
        assert result[2].confidence == 0.5  # unmatched

    def test_empty_proven_changes_nothing(self):
        edges = [_edge("a", "p", "b", 0.7)]
        result = stiffen_edges(edges, [])
        assert result[0].confidence == 0.7


# ── stiffen_to_limit ────────────────────────────────────────────────────────

class TestStiffenToLimit:
    def test_matched_edge_stiffened_to_limit(self):
        edges = [_edge("a", "p", "b", 0.3)]
        proven = [_unit_clause("p", "a", "b")]
        result = stiffen_to_limit(edges, proven, limit=0.75)
        assert result[0].confidence == 0.75

    def test_edge_above_limit_unchanged(self):
        """The limit is a ceiling, not a floor."""
        edges = [_edge("a", "p", "b", 0.9)]
        proven = [_unit_clause("p", "a", "b")]
        result = stiffen_to_limit(edges, proven, limit=0.75)
        assert result[0].confidence == 0.9  # already above limit

    def test_unmatched_edge_unchanged(self):
        edges = [_edge("a", "p", "b", 0.3)]
        proven = [_unit_clause("q", "c", "d")]
        result = stiffen_to_limit(edges, proven, limit=0.75)
        assert result[0].confidence == 0.3

    def test_limit_one_equivalent_to_stiffen(self):
        """stiffen_to_limit with limit=1.0 should match stiffen_edges."""
        edges = [_edge("a", "p", "b", 0.3)]
        proven = [_unit_clause("p", "a", "b")]
        r1 = stiffen_edges(edges, proven)
        r2 = stiffen_to_limit(edges, proven, limit=1.0)
        assert r1[0].confidence == r2[0].confidence


# ── Property-based tests ────────────────────────────────────────────────────

names = st.text(alphabet="abcdefghijklmnop", min_size=1, max_size=4)
confidences = st.floats(min_value=0.01, max_value=0.99)


class TestBridgeProperties:

    @given(names, names, names, confidences)
    def test_clause_from_edge_always_unit(self, subj, pred, obj, conf):
        """clause_from_edge always produces a single-literal clause."""
        edge = _edge(subj, pred, obj, conf)
        clause = clause_from_edge(edge)
        assert len(clause.literals) == 1

    @given(names, names, names, confidences)
    def test_round_trip_preserves_structure(self, subj, pred, obj, conf):
        """edge → clause → edge preserves subject, predicate, object."""
        original = _edge(subj, pred, obj, conf)
        clause = clause_from_edge(original)
        recovered = edge_from_clause(clause)
        assert recovered is not None
        assert recovered.subject == original.subject
        assert recovered.predicate == original.predicate
        assert recovered.object == original.object

    @given(names, names, names, confidences)
    def test_stiffen_raises_confidence(self, subj, pred, obj, conf):
        """Stiffening a matched edge always raises (or maintains) confidence."""
        edges = [_edge(subj, pred, obj, conf)]
        proven = [_unit_clause(pred, subj, obj)]
        result = stiffen_edges(edges, proven)
        assert result[0].confidence >= conf

    @given(names, names, names, confidences,
           st.floats(min_value=0.01, max_value=0.99))
    def test_stiffen_to_limit_bounded(self, subj, pred, obj, conf, limit):
        """After stiffen_to_limit, confidence is at most max(original, limit)."""
        edges = [_edge(subj, pred, obj, conf)]
        proven = [_unit_clause(pred, subj, obj)]
        result = stiffen_to_limit(edges, proven, limit=limit)
        assert result[0].confidence <= max(conf, limit) + 1e-9
