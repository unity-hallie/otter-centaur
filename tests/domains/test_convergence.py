"""
Integration tests for the convergence theorem suite.

All 5 provable theorems must prove. RH must fail.
This is the designed outcome: the system can prove convergence
structure but cannot prove the symmetry of the zeros.
"""

import pytest

from otter.causal_calculus import (
    run_convergence_proof_suite,
    CONVERGENCE_THEOREMS,
    CONVERGENCE_RULES,
    convergence_prune,
)
from otter.core.state import Clause


PROVABLE_THEOREMS = [
    "monotone_bounded",
    "limit_is_reciprocal_zeta",
    "rate_bound",
    "encoding_completeness",
    "fixed_point_existence",
]

WALL_THEOREMS = [
    "rh_symmetry",
]


class TestConvergenceProvableTheorems:

    @pytest.mark.parametrize("theorem_name", PROVABLE_THEOREMS)
    def test_theorem_is_provable(self, theorem_name):
        results = run_convergence_proof_suite(max_steps=100, verbose=False)
        r = results[theorem_name]
        assert r["proved"], (
            f"Theorem '{theorem_name}' not proved in {r['steps']} steps.\n"
            f"Description: {r['description']}"
        )


class TestConvergenceWall:

    @pytest.mark.parametrize("theorem_name", WALL_THEOREMS)
    def test_rh_is_not_provable(self, theorem_name):
        """
        The Riemann Hypothesis should NOT be provable.
        The gap between has_nontrivial_zeros and zeros_on_critical_line
        has no bridging rule. This is the designed outcome.
        """
        results = run_convergence_proof_suite(max_steps=100, verbose=False)
        r = results[theorem_name]
        assert not r["proved"], (
            f"Theorem '{theorem_name}' should NOT be provable, but it was!\n"
            f"This would mean RH has been proved, which is... unlikely."
        )


class TestConvergenceAllTheorems:

    def test_six_theorems_total(self):
        assert len(CONVERGENCE_THEOREMS) == 6

    def test_five_prove_one_fails(self):
        results = run_convergence_proof_suite(max_steps=100, verbose=False)
        proved = [n for n, r in results.items() if r["proved"]]
        failed = [n for n, r in results.items() if not r["proved"]]
        assert len(proved) == 5
        assert len(failed) == 1
        assert failed[0] == "rh_symmetry"


class TestConvergenceRules:

    def test_all_rules_are_clauses(self):
        for rule in CONVERGENCE_RULES:
            assert isinstance(rule, Clause)

    def test_rh_gap_exists(self):
        """
        There should be a rule deriving has_nontrivial_zeros
        and a rule consuming zeros_on_critical_line,
        but NO rule connecting them.
        """
        produces_nontrivial = False
        consumes_critical = False
        bridges_gap = False

        for rule in CONVERGENCE_RULES:
            for lit in rule.literals:
                if lit[0] and lit[1] == "has_nontrivial_zeros":
                    produces_nontrivial = True
                if not lit[0] and lit[1] == "zeros_on_critical_line":
                    consumes_critical = True
            # Check if any single rule has both
            preds_pos = {lit[1] for lit in rule.literals if lit[0]}
            preds_neg = {lit[1] for lit in rule.literals if not lit[0]}
            if "has_nontrivial_zeros" in preds_neg and "zeros_on_critical_line" in preds_pos:
                bridges_gap = True

        assert produces_nontrivial, "Should have rule producing has_nontrivial_zeros"
        assert consumes_critical, "Should have rule consuming zeros_on_critical_line"
        assert not bridges_gap, "The gap must NOT be bridged"

    def test_shortcut_fp_exists(self):
        """The SHORTCUT-FP axiom should be present."""
        labels = [r.label for r in CONVERGENCE_RULES]
        assert any("SHORTCUT-FP" in l for l in labels)


class TestConvergencePrune:

    def test_prunes_large_clause(self):
        large = Clause(literals=frozenset({
            (True, "a", "x"), (True, "b", "x"), (True, "c", "x"),
            (True, "d", "x"), (True, "e", "x"), (True, "f", "x"),
        }))
        assert convergence_prune(large, None)

    def test_does_not_prune_small_clause(self):
        small = Clause(literals=frozenset({(True, "a", "x")}))
        assert not convergence_prune(small, None)

    def test_does_not_prune_non_clause(self):
        from otter.core.state import Item
        item = Item(name="test", content="test")
        assert not convergence_prune(item, None)
