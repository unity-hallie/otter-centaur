"""
Unit tests for the zeta evidence stream and Euler product.

Core claims:
    - zeta_evidence_stream generates edges with confidence 1 - p^{-s}
    - Confidences are monotone increasing (larger primes → higher confidence)
    - All confidences are in (0, 1)
    - The partial product ∏(1 - p^{-s}) is the reciprocal of the partial ζ
    - _first_n_primes returns the correct primes
    - run_zeta_approach produces a ConvergentProof with the right structure
"""

import pytest
import math
from hypothesis import given, assume
from hypothesis import strategies as st

from otter.causal_calculus import (
    zeta_evidence_stream, zeta_partial_product,
    _first_n_primes, run_zeta_approach,
    ConvergentProof,
)


# ── Unit tests: _first_n_primes ─────────────────────────────────────────────

class TestFirstNPrimes:
    def test_first_five(self):
        assert _first_n_primes(5) == [2, 3, 5, 7, 11]

    def test_first_one(self):
        assert _first_n_primes(1) == [2]

    def test_first_zero(self):
        assert _first_n_primes(0) == []

    def test_first_ten(self):
        expected = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        assert _first_n_primes(10) == expected


# ── Unit tests: zeta_evidence_stream ─────────────────────────────────────────

class TestZetaEvidenceStream:
    def test_correct_number_of_edges(self):
        stream = zeta_evidence_stream(s=2.0, num_primes=5)
        assert len(stream) == 5

    def test_confidences_match_formula(self):
        """Edge confidence should be exactly 1 - p^{-s}."""
        stream = zeta_evidence_stream(s=2.0, num_primes=5)
        primes = [2, 3, 5, 7, 11]
        for edge, p in zip(stream, primes):
            expected = 1.0 - p ** (-2.0)
            assert edge.confidence == pytest.approx(expected)

    def test_confidences_are_monotone_increasing(self):
        """Larger primes → less uncertainty → higher confidence."""
        stream = zeta_evidence_stream(s=2.0, num_primes=10)
        confs = [e.confidence for e in stream]
        for i in range(len(confs) - 1):
            assert confs[i] < confs[i+1]

    def test_all_confidences_in_unit_interval(self):
        stream = zeta_evidence_stream(s=2.0, num_primes=15)
        for edge in stream:
            assert 0 < edge.confidence < 1

    def test_edge_predicate_is_contributes(self):
        stream = zeta_evidence_stream(s=2.0, num_primes=3)
        for edge in stream:
            assert edge.predicate == "contributes_to_encoding"

    def test_edge_subject_names_prime(self):
        stream = zeta_evidence_stream(s=2.0, num_primes=3)
        assert stream[0].subject == "prime_2"
        assert stream[1].subject == "prime_3"
        assert stream[2].subject == "prime_5"


# ── Unit tests: zeta_partial_product ─────────────────────────────────────────

class TestZetaPartialProduct:
    def test_known_value_s2(self):
        """ζ(2) = π²/6. Partial product with many primes should approach this."""
        primes = _first_n_primes(50)
        partial = zeta_partial_product(2.0, primes)
        assert partial == pytest.approx(math.pi**2 / 6, rel=0.01)

    def test_single_prime(self):
        """With just prime 2 at s=2: 1/(1 - 1/4) = 4/3."""
        result = zeta_partial_product(2.0, [2])
        assert result == pytest.approx(4.0 / 3.0)

    def test_reciprocal_relationship(self):
        """conf_N × ζ_N = 1 exactly."""
        primes = _first_n_primes(10)
        for n in range(1, len(primes) + 1):
            p_subset = primes[:n]
            zeta_n = zeta_partial_product(2.0, p_subset)
            conf_n = 1.0
            for p in p_subset:
                conf_n *= (1.0 - p ** (-2.0))
            assert conf_n * zeta_n == pytest.approx(1.0)


# ── Unit tests: run_zeta_approach ────────────────────────────────────────────

class TestRunZetaApproach:
    def test_returns_convergent_proof(self):
        result = run_zeta_approach(s=2.0, num_primes=5, verbose=False)
        assert isinstance(result, ConvergentProof)

    def test_correct_number_of_steps(self):
        result = run_zeta_approach(s=2.0, num_primes=5, verbose=False)
        assert len(result.steps) == 5

    def test_all_steps_proved(self):
        """Every step should produce a proof (the rule fires on any single prime)."""
        result = run_zeta_approach(s=2.0, num_primes=5, verbose=False)
        for label, cp in result.steps:
            assert cp is not None

    def test_confidences_are_euler_product(self):
        """The confidence at step N should be ∏_{i=1}^{N} (1 - p_i^{-s})."""
        result = run_zeta_approach(s=2.0, num_primes=5, verbose=False)
        primes = _first_n_primes(5)
        running = 1.0
        for i, (label, cp) in enumerate(result.steps):
            running *= (1.0 - primes[i] ** (-2.0))
            assert cp.conditional_confidence == pytest.approx(running)

    def test_is_monotone_decreasing(self):
        """The Euler product is monotone decreasing (each factor < 1)."""
        result = run_zeta_approach(s=2.0, num_primes=10, verbose=False)
        assert result.is_monotone()
        assert result.monotone_direction == "decreasing"


# ── Property-based tests ────────────────────────────────────────────────────

class TestZetaProperties:

    @given(st.floats(min_value=1.1, max_value=10.0),
           st.integers(min_value=1, max_value=15))
    def test_all_confidences_subunit(self, s, n):
        """For s > 1, every edge confidence is in (0, 1].

        Mathematically strictly < 1, but at high s, p^{-s} underflows
        to 0.0 in float64, making 1 - p^{-s} = 1.0 exactly.
        """
        stream = zeta_evidence_stream(s=s, num_primes=n)
        for edge in stream:
            assert 0 < edge.confidence <= 1

    @given(st.floats(min_value=1.1, max_value=10.0),
           st.integers(min_value=2, max_value=15))
    def test_confidences_monotone(self, s, n):
        """Larger primes always have >= confidence (for any s > 1).

        At high s, p^{-s} is so small that 1 - p^{-s} ≈ 1.0 for all
        large primes, making consecutive confidences equal in float.
        """
        stream = zeta_evidence_stream(s=s, num_primes=n)
        confs = [e.confidence for e in stream]
        for i in range(len(confs) - 1):
            assert confs[i] <= confs[i+1]

    @given(st.floats(min_value=1.1, max_value=10.0),
           st.integers(min_value=1, max_value=10))
    def test_reciprocal_identity(self, s, n):
        """conf_N × ζ_N = 1 for any s > 1 and any N."""
        primes = _first_n_primes(n)
        zeta_n = zeta_partial_product(s, primes)
        conf_n = 1.0
        for p in primes:
            conf_n *= (1.0 - p ** (-s))
        assert conf_n * zeta_n == pytest.approx(1.0, rel=1e-9)

    @given(st.integers(min_value=1, max_value=20))
    def test_primes_are_prime(self, n):
        """Every number returned by _first_n_primes is actually prime."""
        primes = _first_n_primes(n)
        for p in primes:
            assert p >= 2
            for d in range(2, int(p**0.5) + 1):
                assert p % d != 0
