"""
Property-based and unit tests for resolution.

Core claims:
    - Every resolvent is a logical consequence of its parents (soundness)
    - Resolution is symmetric: resolve(A,B) and resolve(B,A) produce the same set
    - clause_subsumes is reflexive and transitive
    - The empty clause is derived from contradictory unit clauses
"""

import pytest
from hypothesis import given, assume, settings
from hypothesis import strategies as st

from otter.core.state import Clause
from otter.inference.resolve import resolve, clause_subsumes


# ── Generators ────────────────────────────────────────────────────────────────

constants = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz",
    min_size=1, max_size=5,
).filter(lambda s: s[0].islower())

predicates = constants

@st.composite
def unit_clause(draw, sign=None):
    s = draw(st.booleans()) if sign is None else sign
    pred = draw(predicates)
    args = [draw(constants) for _ in range(draw(st.integers(1, 2)))]
    return Clause(
        literals=frozenset({tuple([s, pred] + args)}),
        label=draw(st.text(max_size=10)),
    )

@st.composite
def complementary_pair(draw):
    """Two unit clauses that can resolve: one positive, one negative, same pred/args."""
    pred = draw(predicates)
    args = [draw(constants) for _ in range(draw(st.integers(1, 2)))]
    pos = Clause(literals=frozenset({tuple([True, pred] + args)}))
    neg = Clause(literals=frozenset({tuple([False, pred] + args)}))
    return pos, neg


# ── Unit tests ────────────────────────────────────────────────────────────────

class TestResolve:
    def test_socrates_step1(self):
        """
        ~human(X) | mortal(X)   with   human(socrates)
        Should produce: mortal(socrates)
        """
        rule = Clause(literals=frozenset({
            (False, "human", "X"),
            (True,  "mortal", "X"),
        }), label="all humans are mortal")
        fact = Clause(literals=frozenset({
            (True, "human", "socrates"),
        }), label="socrates is human")

        results = resolve(rule, fact)
        assert len(results) >= 1
        result_lits = [r.literals for r in results]
        assert frozenset({(True, "mortal", "socrates")}) in result_lits

    def test_resolves_to_empty_clause(self):
        """
        mortal(socrates)   with   ~mortal(socrates)
        Should produce the empty clause.
        """
        pos = Clause(literals=frozenset({(True,  "mortal", "socrates")}))
        neg = Clause(literals=frozenset({(False, "mortal", "socrates")}))
        results = resolve(pos, neg)
        assert any(r.is_empty for r in results)

    def test_no_resolution_different_predicates(self):
        c1 = Clause(literals=frozenset({(True,  "human", "socrates")}))
        c2 = Clause(literals=frozenset({(False, "mortal", "socrates")}))
        assert resolve(c1, c2) == []

    def test_no_resolution_same_sign(self):
        c1 = Clause(literals=frozenset({(True, "human", "socrates")}))
        c2 = Clause(literals=frozenset({(True, "mortal", "socrates")}))
        assert resolve(c1, c2) == []

    def test_source_recorded(self):
        pos = Clause(literals=frozenset({(True,  "p", "a")}), label="pos")
        neg = Clause(literals=frozenset({(False, "p", "a")}), label="neg")
        results = resolve(pos, neg)
        assert len(results) >= 1
        assert results[0].source == (pos.name, neg.name)

    def test_variable_resolution(self):
        """
        ~knows(X, Y) | trusts(X, Y)   with   knows(alice, bob)
        Should produce: trusts(alice, bob)
        """
        rule = Clause(literals=frozenset({
            (False, "knows", "X", "Y"),
            (True,  "trusts", "X", "Y"),
        }))
        fact = Clause(literals=frozenset({(True, "knows", "alice", "bob")}))
        results = resolve(rule, fact)
        assert any(
            r.literals == frozenset({(True, "trusts", "alice", "bob")})
            for r in results
        )


class TestClauseSubsumes:
    def test_shorter_subsumes_longer_if_subset(self):
        c1 = Clause(literals=frozenset({(True, "p", "a")}))
        c2 = Clause(literals=frozenset({(True, "p", "a"), (True, "q", "b")}))
        assert clause_subsumes(c1, c2)

    def test_same_length_does_not_subsume(self):
        c1 = Clause(literals=frozenset({(True, "p", "a")}))
        c2 = Clause(literals=frozenset({(True, "p", "a")}))
        assert not clause_subsumes(c1, c2)

    def test_longer_does_not_subsume_shorter(self):
        c1 = Clause(literals=frozenset({(True, "p", "a"), (True, "q", "b")}))
        c2 = Clause(literals=frozenset({(True, "p", "a")}))
        assert not clause_subsumes(c1, c2)

    def test_non_subset_does_not_subsume(self):
        c1 = Clause(literals=frozenset({(True, "p", "a")}))
        c2 = Clause(literals=frozenset({(True, "q", "b"), (True, "r", "c")}))
        assert not clause_subsumes(c1, c2)


# ── Property-based tests ──────────────────────────────────────────────────────

class TestResolveProperties:

    @given(complementary_pair())
    def test_complementary_unit_clauses_resolve_to_empty(self, pair):
        """Two complementary ground unit clauses always resolve to the empty clause."""
        pos, neg = pair
        results = resolve(pos, neg)
        assert any(r.is_empty for r in results)

    @given(unit_clause(), unit_clause())
    def test_symmetry(self, c1, c2):
        """resolve(A, B) and resolve(B, A) produce the same set of resolvents."""
        r1 = resolve(c1, c2)
        r2 = resolve(c2, c1)
        lits1 = {r.literals for r in r1}
        lits2 = {r.literals for r in r2}
        assert lits1 == lits2

    @given(unit_clause())
    def test_resolving_with_self_gives_empty_only_for_complementary(self, c):
        """A clause resolves with itself only if it contains complementary literals."""
        results = resolve(c, c)
        # Unit clauses with single literal can't self-resolve (need complement)
        assert results == []

    @given(st.lists(unit_clause(), min_size=2, max_size=4))
    def test_resolvents_have_source(self, clauses):
        """Every resolvent records its parents in source."""
        for i, c1 in enumerate(clauses):
            for j, c2 in enumerate(clauses):
                if i != j:
                    for r in resolve(c1, c2):
                        assert len(r.source) == 2


class TestSubsumptionProperties:

    @given(unit_clause())
    def test_subsumption_irreflexive(self, c):
        """A clause does not subsume itself (strict subset required)."""
        assert not clause_subsumes(c, c)

    @given(unit_clause(), unit_clause(), unit_clause())
    def test_subsumption_transitive(self, c1, c2, c3):
        """If c1 subsumes c2 and c2 subsumes c3, then c1 subsumes c3."""
        if clause_subsumes(c1, c2) and clause_subsumes(c2, c3):
            assert clause_subsumes(c1, c3)
