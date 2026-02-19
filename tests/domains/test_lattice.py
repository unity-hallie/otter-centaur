"""
Integration tests for the prime factor lattice domain.

All 18 theorems (11 structure + 7 wave) must be provable.
"""

import pytest

from otter.domains.lattice import (
    run_lattice_proof_suite,
    make_lattice_state,
    LATTICE_THEOREMS,
)


STRUCTURE_THEOREMS = [
    "reflexivity", "antisymmetry", "transitivity", "unit_is_bottom",
    "gcd_is_lower_bound", "gcd_is_greatest", "lcm_is_upper_bound",
    "lcm_is_least", "gcd_is_meet", "lcm_is_join", "distributivity",
]

WAVE_THEOREMS = [
    "prime_density", "prime_independence", "density_product",
    "prime_wave", "wave_orthogonality", "wave_basis", "fta_as_probability",
]


class TestLatticeStructureTheorems:

    @pytest.mark.parametrize("theorem_name", STRUCTURE_THEOREMS)
    def test_structure_theorem_provable(self, theorem_name):
        results = run_lattice_proof_suite(max_steps=100, verbose=False)
        r = results[theorem_name]
        assert r["proved"], (
            f"Structure theorem '{theorem_name}' not proved in {r['steps']} steps.\n"
            f"Description: {r['description']}"
        )


class TestLatticeWaveTheorems:

    @pytest.mark.parametrize("theorem_name", WAVE_THEOREMS)
    def test_wave_theorem_provable(self, theorem_name):
        results = run_lattice_proof_suite(max_steps=100, verbose=False)
        r = results[theorem_name]
        assert r["proved"], (
            f"Wave theorem '{theorem_name}' not proved in {r['steps']} steps.\n"
            f"Description: {r['description']}"
        )


class TestLatticeAllTheorems:

    def test_all_18_theorems_proved(self):
        results = run_lattice_proof_suite(max_steps=100, verbose=False)
        assert len(results) == 18, f"Expected 18 theorems, got {len(results)}"
        failed = [name for name, r in results.items() if not r["proved"]]
        assert failed == [], f"Failed theorems: {failed}"


class TestLatticeStateSetup:

    def test_make_state_unknown_theorem_raises(self):
        with pytest.raises(ValueError, match="Unknown theorem"):
            make_lattice_state(theorem="not_a_theorem")

    def test_default_theorem_is_reflexivity(self):
        state = make_lattice_state()
        labels = {item.label for item in state.set_of_support}
        assert any("DIV-REFL" in l for l in labels)

    @pytest.mark.parametrize("theorem_name", list(LATTICE_THEOREMS.keys()))
    def test_state_has_negated_goal(self, theorem_name):
        state = make_lattice_state(theorem=theorem_name)
        thm = LATTICE_THEOREMS[theorem_name]
        negated_goal = thm["negated_goal"]
        assert negated_goal in list(state.set_of_support)
