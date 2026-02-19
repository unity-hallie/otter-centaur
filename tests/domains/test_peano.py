"""
Integration tests for the Peano arithmetic domain.

These are integration-level (not property-based) because the claims
are axiom-specific: "1+1=2 is provable in the Peano system."
"""

import pytest

from otter.core.engine import run_otter
from otter.core.proof import found_empty_clause, extract_proof
from otter.inference.paramodulate import resolve_and_paramodulate
from otter.inference.resolve import clause_subsumes
from otter.domains.peano import make_peano_state, peano_prune, PEANO_RULES
from otter.core.state import Clause


class TestPeanoProves1Plus1Equals2:
    def test_proves_in_bounded_steps(self):
        state = make_peano_state()
        state = run_otter(
            state, resolve_and_paramodulate,
            max_steps=200,
            stop_fn=found_empty_clause,
            subsumes_fn=clause_subsumes,
            prune_fn=peano_prune,
            verbose=False,
        )
        assert found_empty_clause(state), (
            f"1+1=2 not proved in {state.step} steps"
        )

    def test_proof_is_extractable(self):
        state = make_peano_state()
        state = run_otter(
            state, resolve_and_paramodulate,
            max_steps=200,
            stop_fn=found_empty_clause,
            subsumes_fn=clause_subsumes,
            prune_fn=peano_prune,
            verbose=False,
        )
        if found_empty_clause(state):
            proof = extract_proof(state)
            assert len(proof) > 0
            # Last step should be the empty clause
            last_clause, _ = proof[-1]
            assert last_clause.is_empty

    def test_proof_ends_with_empty_clause(self):
        state = make_peano_state()
        state = run_otter(
            state, resolve_and_paramodulate,
            max_steps=200,
            stop_fn=found_empty_clause,
            subsumes_fn=clause_subsumes,
            prune_fn=peano_prune,
            verbose=False,
        )
        if found_empty_clause(state):
            all_items = list(state.set_of_support) + state.usable
            empty_clauses = [c for c in all_items
                             if isinstance(c, Clause) and c.is_empty]
            assert len(empty_clauses) >= 1


class TestPeanoPrune:
    def test_prunes_deep_terms(self):
        # A clause with depth > 6 should be pruned
        deep_term = "0"
        for _ in range(7):
            deep_term = ("s", deep_term)
        deep_clause = Clause(
            literals=frozenset({(True, "nat", deep_term)}),
        )
        assert peano_prune(deep_clause, None)

    def test_does_not_prune_shallow_terms(self):
        # s(s(0)) has depth 2 -- fine
        shallow = Clause(
            literals=frozenset({(True, "nat", ("s", ("s", "0")))}),
        )
        assert not peano_prune(shallow, None)

    def test_does_not_prune_non_clause(self):
        from otter.core.state import Item
        item = Item(name="test", content="test")
        assert not peano_prune(item, None)


class TestPeanoRules:
    def test_all_rules_are_clauses(self):
        for rule in PEANO_RULES:
            assert isinstance(rule, Clause)

    def test_pa1_is_unit_clause(self):
        pa1 = next(r for r in PEANO_RULES if r.label == "PA1: zero is nat")
        assert len(pa1.literals) == 1

    def test_has_expected_axiom_count(self):
        assert len(PEANO_RULES) == 9  # PA1-PA8 + reflexivity
