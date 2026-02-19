"""
Unit tests for the self-referential convergence loop.

Core claims:
    - encode_proof_steps produces a prime factorization dict
    - The factorization always contains primes 2, 3, 5 (structural floor)
    - factorize_to_evidence converts a factorization back to an evidence stream
    - The stream is sorted by descending exponent, then ascending prime
    - self_referential_convergence reaches a fixed point
    - The fixed point stabilizes within a small number of iterations
"""

import pytest

from otter.causal_calculus import (
    encode_proof_steps, factorize_to_evidence,
    self_referential_convergence, run_zeta_approach,
    _first_n_primes, ConvergentProof,
)
from otter.conditional_proof import ConditionalProof


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_initial_proof():
    """Get a ConvergentProof from the zeta approach for encoding tests."""
    return run_zeta_approach(s=2.0, num_primes=5, verbose=False)


# ── Unit tests: encode_proof_steps ──────────────────────────────────────────

class TestEncodeProofSteps:
    def test_returns_dict(self):
        proof = _get_initial_proof()
        fact = encode_proof_steps(proof)
        assert isinstance(fact, dict)

    def test_keys_are_primes(self):
        proof = _get_initial_proof()
        fact = encode_proof_steps(proof)
        primes_set = set(_first_n_primes(50))
        for key in fact:
            assert key in primes_set, f"{key} is not prime"

    def test_values_are_positive_integers(self):
        proof = _get_initial_proof()
        fact = encode_proof_steps(proof)
        for v in fact.values():
            assert isinstance(v, int)
            assert v > 0

    def test_structural_primes_always_present(self):
        """Every encoding includes {2, 3, 5} as the structural floor."""
        proof = _get_initial_proof()
        fact = encode_proof_steps(proof)
        assert 2 in fact
        assert 3 in fact
        assert 5 in fact

    def test_empty_proof_still_has_structural_primes(self):
        """Even a proof with no proved steps has the {2,3,5} floor."""
        empty = ConvergentProof(conclusion="test", steps=[])
        fact = encode_proof_steps(empty)
        assert 2 in fact
        assert 3 in fact
        assert 5 in fact


# ── Unit tests: factorize_to_evidence ────────────────────────────────────────

class TestFactorizeToEvidence:
    def test_returns_edge_list(self):
        fact = {2: 5, 3: 3, 5: 1}
        stream = factorize_to_evidence(fact, s=2.0)
        assert len(stream) == 3

    def test_edge_confidences_match_formula(self):
        """Edge confidence = 1 - p^{-s}."""
        fact = {2: 1, 7: 1}
        stream = factorize_to_evidence(fact, s=2.0)
        for edge in stream:
            p = int(edge.subject.split("_")[1])
            expected = 1.0 - p ** (-2.0)
            assert edge.confidence == pytest.approx(expected)

    def test_sorted_by_exponent_then_prime(self):
        """Higher exponent first; within same exponent, smaller prime first."""
        fact = {5: 2, 2: 5, 3: 5, 7: 1}
        stream = factorize_to_evidence(fact, s=2.0)
        # 2 (exp 5), 3 (exp 5), 5 (exp 2), 7 (exp 1)
        primes = [int(e.subject.split("_")[1]) for e in stream]
        assert primes == [2, 3, 5, 7]

    def test_predicate_is_encodes_proof_structure(self):
        fact = {2: 1}
        stream = factorize_to_evidence(fact, s=2.0)
        assert stream[0].predicate == "encodes_proof_structure"

    def test_empty_factorization_gives_empty_stream(self):
        stream = factorize_to_evidence({}, s=2.0)
        assert stream == []


# ── Unit tests: self_referential_convergence ─────────────────────────────────

class TestSelfReferentialConvergence:
    def test_returns_list_of_tuples(self):
        iterations = self_referential_convergence(
            s=2.0, num_primes_initial=5, max_iterations=5,
            epsilon=1e-4, verbose=False,
        )
        assert isinstance(iterations, list)
        assert len(iterations) > 0
        for item in iterations:
            assert len(item) == 3  # (iteration_num, ConvergentProof, factorization)

    def test_first_iteration_is_zero(self):
        iterations = self_referential_convergence(
            s=2.0, num_primes_initial=5, max_iterations=3,
            epsilon=1e-4, verbose=False,
        )
        assert iterations[0][0] == 0  # iteration number

    def test_reaches_fixed_point(self):
        """The iteration should converge within 10 iterations at s=2."""
        iterations = self_referential_convergence(
            s=2.0, num_primes_initial=12, max_iterations=10,
            epsilon=1e-6, verbose=False,
        )
        # Should converge before max_iterations
        assert len(iterations) < 10

    def test_limits_stabilize(self):
        """The final two iterations should have very close limits."""
        iterations = self_referential_convergence(
            s=2.0, num_primes_initial=12, max_iterations=10,
            epsilon=1e-6, verbose=False,
        )
        if len(iterations) >= 2:
            lim_last = iterations[-1][1].limit
            lim_prev = iterations[-2][1].limit
            assert lim_last is not None
            assert lim_prev is not None
            assert abs(lim_last - lim_prev) < 1e-4

    def test_primes_stabilize(self):
        """The set of primes in the encoding should stabilize."""
        iterations = self_referential_convergence(
            s=2.0, num_primes_initial=12, max_iterations=10,
            epsilon=1e-6, verbose=False,
        )
        if len(iterations) >= 2:
            primes_last = set(iterations[-1][2].keys())
            primes_prev = set(iterations[-2][2].keys())
            assert primes_last == primes_prev

    def test_each_iteration_produces_convergent_proof(self):
        iterations = self_referential_convergence(
            s=2.0, num_primes_initial=5, max_iterations=3,
            epsilon=1e-4, verbose=False,
        )
        for _, proof, _ in iterations:
            assert isinstance(proof, ConvergentProof)

    def test_each_iteration_produces_factorization(self):
        iterations = self_referential_convergence(
            s=2.0, num_primes_initial=5, max_iterations=3,
            epsilon=1e-4, verbose=False,
        )
        for _, _, fact in iterations:
            assert isinstance(fact, dict)
            assert all(isinstance(k, int) for k in fact)
            assert all(isinstance(v, int) for v in fact.values())
