"""
Integration tests for the Gödel numbering domain.

All 8 theorems must be provable. These run the full proof suites.
"""

import pytest

from otter.domains.goedel import (
    run_goedel_proof_suite,
    make_goedel_state,
    GOEDEL_THEOREMS,
    goedel_symbol_table,
    verify_symbol_coverage,
)
from otter.domains.peano import PEANO_RULES


class TestGoedelTheoremSuite:

    @pytest.mark.parametrize("theorem_name", list(GOEDEL_THEOREMS.keys()))
    def test_theorem_is_provable(self, theorem_name):
        results = run_goedel_proof_suite(max_steps=50, verbose=False)
        r = results[theorem_name]
        assert r["proved"], (
            f"Theorem '{theorem_name}' not proved in {r['steps']} steps.\n"
            f"Description: {r['description']}"
        )

    def test_all_theorems_proved(self):
        results = run_goedel_proof_suite(max_steps=50, verbose=False)
        failed = [name for name, r in results.items() if not r["proved"]]
        assert failed == [], f"Failed theorems: {failed}"


class TestGoedelSymbolCoverage:

    def test_peano_symbols_covered(self):
        table = goedel_symbol_table()
        covered, missing = verify_symbol_coverage(PEANO_RULES, table)
        assert missing == set(), (
            f"Peano axiom symbols not in Goedel table: {missing}"
        )

    def test_symbol_table_has_variables(self):
        table = goedel_symbol_table()
        assert "X" in table
        assert "Y" in table

    def test_symbol_table_has_peano_symbols(self):
        table = goedel_symbol_table()
        for sym in ["0", "s", "plus", "times", "eq", "nat"]:
            assert sym in table, f"Missing: {sym}"


class TestGoedelStateSetup:

    def test_make_state_unknown_theorem_raises(self):
        with pytest.raises(ValueError, match="Unknown theorem"):
            make_goedel_state(theorem="not_a_theorem")

    def test_default_theorem_is_injectivity(self):
        state = make_goedel_state()
        # Injectivity uses G-PROD, FTA, G-SEQ, G-INJ axioms
        labels = {item.label for item in state.set_of_support}
        assert any("G-PROD" in l for l in labels)

    @pytest.mark.parametrize("theorem_name", list(GOEDEL_THEOREMS.keys()))
    def test_state_has_negated_goal(self, theorem_name):
        state = make_goedel_state(theorem=theorem_name)
        thm = GOEDEL_THEOREMS[theorem_name]
        negated_goal = thm["negated_goal"]
        assert negated_goal in list(state.set_of_support)
