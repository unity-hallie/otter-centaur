"""
Property-based and unit tests for ConvergentProof and converge_conditionally.

Core claims:
    - confidences extracts only proved steps
    - derivative(0) is the first confidence, derivative(i) is the difference
    - integral is the sum of all confidences
    - is_monotone detects both non-decreasing and non-increasing sequences
    - monotone_direction correctly classifies increasing/decreasing/constant
    - is_cauchy detects when the tail is within epsilon
    - limit extrapolates geometrically from the last two deltas
    - converge_conditionally builds a ConvergentProof from a growing evidence prefix
"""

import pytest
import math
from hypothesis import given, assume, settings
from hypothesis import strategies as st

from otter.causal_calculus import (
    ConvergentProof, converge_conditionally,
    zeta_evidence_stream, _first_n_primes,
)
from otter.conditional_proof import ConditionalProof


# ── Helpers ──────────────────────────────────────────────────────────────────

def _cp(conf):
    """Make a minimal ConditionalProof with given confidence."""
    return ConditionalProof(
        conclusion="test(a, b)",
        proof_steps=[],
        axiom_confidences={"ax": conf},
        conditional_confidence=conf,
    )


def _convergent(confs, conclusion="test(a, b)"):
    """Make a ConvergentProof from a list of confidence values."""
    steps = [(f"step_{i}", _cp(c)) for i, c in enumerate(confs)]
    return ConvergentProof(conclusion=conclusion, steps=steps)


def _convergent_with_nones(items, conclusion="test(a, b)"):
    """Make a ConvergentProof where items are (label, conf_or_None)."""
    steps = []
    for label, conf in items:
        if conf is None:
            steps.append((label, None))
        else:
            steps.append((label, _cp(conf)))
    return ConvergentProof(conclusion=conclusion, steps=steps)


# ── Unit tests: confidences ──────────────────────────────────────────────────

class TestConfidences:
    def test_extracts_all_proved_steps(self):
        cp = _convergent([0.5, 0.6, 0.7])
        confs = cp.confidences
        assert len(confs) == 3
        assert [c for _, c in confs] == [0.5, 0.6, 0.7]

    def test_skips_none_steps(self):
        cp = _convergent_with_nones([
            ("a", None), ("b", 0.5), ("c", None), ("d", 0.7),
        ])
        confs = cp.confidences
        assert len(confs) == 2
        assert [c for _, c in confs] == [0.5, 0.7]

    def test_empty_steps(self):
        cp = ConvergentProof(conclusion="test", steps=[])
        assert cp.confidences == []


# ── Unit tests: derivative ───────────────────────────────────────────────────

class TestDerivative:
    def test_derivative_zero_is_first_confidence(self):
        cp = _convergent([0.5, 0.6, 0.7])
        assert cp.derivative(0) == 0.5

    def test_derivative_i_is_difference(self):
        cp = _convergent([0.5, 0.6, 0.75])
        assert cp.derivative(1) == pytest.approx(0.1)
        assert cp.derivative(2) == pytest.approx(0.15)

    def test_derivative_out_of_range_is_none(self):
        cp = _convergent([0.5, 0.6])
        assert cp.derivative(5) is None

    def test_derivative_empty_is_none(self):
        cp = _convergent([])
        assert cp.derivative(0) is None


# ── Unit tests: integral ─────────────────────────────────────────────────────

class TestIntegral:
    def test_integral_is_sum(self):
        cp = _convergent([0.5, 0.6, 0.7])
        assert cp.integral() == pytest.approx(1.8)

    def test_integral_empty_is_zero(self):
        cp = _convergent([])
        assert cp.integral() == 0.0


# ── Unit tests: convergence_rate ─────────────────────────────────────────────

class TestConvergenceRate:
    def test_rate_at_zero_is_one(self):
        """derivative(0) / confidence(0) = c_0 / c_0 = 1.0 always."""
        cp = _convergent([0.5, 0.6, 0.7])
        assert cp.convergence_rate(0) == pytest.approx(1.0)

    def test_rate_decreases_for_convergent_sequence(self):
        """For a converging sequence, the rate should decrease."""
        cp = _convergent([0.5, 0.7, 0.8, 0.85])
        rates = [cp.convergence_rate(i) for i in range(4)]
        # Rate should be decreasing after step 0
        for i in range(2, len(rates)):
            assert rates[i] < rates[i-1]

    def test_rate_out_of_range_is_none(self):
        cp = _convergent([0.5])
        assert cp.convergence_rate(5) is None


# ── Unit tests: monotonicity ────────────────────────────────────────────────

class TestMonotone:
    def test_increasing_is_monotone(self):
        cp = _convergent([0.3, 0.5, 0.7, 0.8])
        assert cp.is_monotone()
        assert cp.monotone_direction == "increasing"

    def test_decreasing_is_monotone(self):
        cp = _convergent([0.9, 0.7, 0.5, 0.3])
        assert cp.is_monotone()
        assert cp.monotone_direction == "decreasing"

    def test_constant_is_monotone(self):
        cp = _convergent([0.5, 0.5, 0.5])
        assert cp.is_monotone()
        assert cp.monotone_direction == "constant"

    def test_oscillating_is_not_monotone(self):
        cp = _convergent([0.3, 0.7, 0.4, 0.8])
        assert not cp.is_monotone()
        assert cp.monotone_direction is None

    def test_single_step_is_monotone(self):
        cp = _convergent([0.5])
        assert cp.is_monotone()

    def test_empty_is_monotone(self):
        cp = _convergent([])
        assert cp.is_monotone()


# ── Unit tests: is_cauchy ────────────────────────────────────────────────────

class TestIsCauchy:
    def test_converged_sequence_is_cauchy(self):
        cp = _convergent([0.5, 0.6, 0.605, 0.607, 0.6075, 0.6078, 0.6079, 0.6079])
        assert cp.is_cauchy(epsilon=0.01)

    def test_divergent_sequence_is_not_cauchy(self):
        cp = _convergent([0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 0.5, 0.1])
        assert not cp.is_cauchy(epsilon=0.01)

    def test_too_few_steps_is_not_cauchy(self):
        """Need at least 4 steps for Cauchy check."""
        cp = _convergent([0.5, 0.5, 0.5])
        assert not cp.is_cauchy()

    def test_strict_epsilon(self):
        cp = _convergent([0.5, 0.6, 0.65, 0.67, 0.68, 0.685, 0.687, 0.688])
        assert cp.is_cauchy(epsilon=0.01)
        assert not cp.is_cauchy(epsilon=0.001)


# ── Unit tests: limit ────────────────────────────────────────────────────────

class TestLimit:
    def test_geometric_sequence_limit(self):
        """For a geometrically converging sequence, limit should extrapolate."""
        # Geometric approach: 1 - (1/2)^n → 1.0
        confs = [1.0 - 0.5**n for n in range(1, 8)]
        cp = _convergent(confs)
        lim = cp.limit
        assert lim is not None
        assert lim > confs[-1]  # extrapolates beyond last value
        assert lim <= 1.0       # bounded

    def test_single_step_limit(self):
        cp = _convergent([0.75])
        assert cp.limit == 0.75

    def test_empty_limit_is_none(self):
        cp = _convergent([])
        assert cp.limit is None

    def test_limit_capped_at_one(self):
        """Limit should never exceed 1.0."""
        confs = [0.9, 0.95, 0.975, 0.9875, 0.99375]
        cp = _convergent(confs)
        lim = cp.limit
        assert lim is not None
        assert lim <= 1.0

    def test_two_step_limit(self):
        cp = _convergent([0.5, 0.7])
        lim = cp.limit
        assert lim is not None
        assert lim >= 0.7  # at least as big as last value


# ── Unit tests: repr ─────────────────────────────────────────────────────────

class TestConvergentProofRepr:
    def test_repr_includes_conclusion(self):
        cp = _convergent([0.5, 0.6])
        assert "test(a, b)" in repr(cp)

    def test_repr_includes_step_count(self):
        cp = _convergent([0.5, 0.6, 0.7])
        assert "3 steps" in repr(cp)


# ── Property-based tests ────────────────────────────────────────────────────

confidence_lists = st.lists(
    st.floats(min_value=0.01, max_value=0.99),
    min_size=1, max_size=10,
)

sorted_confidence_lists = st.lists(
    st.floats(min_value=0.01, max_value=0.99),
    min_size=2, max_size=10,
).map(sorted)


class TestConvergentProofProperties:

    @given(confidence_lists)
    def test_integral_equals_sum_of_confidences(self, confs):
        """Integral is always exactly the sum of confidence values."""
        cp = _convergent(confs)
        assert cp.integral() == pytest.approx(sum(confs))

    @given(confidence_lists)
    def test_derivative_zero_equals_first_confidence(self, confs):
        """The zeroth derivative is always the first confidence."""
        cp = _convergent(confs)
        assert cp.derivative(0) == pytest.approx(confs[0])

    @given(sorted_confidence_lists)
    def test_sorted_sequence_is_monotone(self, confs):
        """A sorted list of confidences is always monotone increasing."""
        cp = _convergent(confs)
        assert cp.is_monotone()

    @given(confidence_lists)
    def test_derivatives_sum_to_last_confidence(self, confs):
        """Sum of all derivatives = last confidence (telescoping)."""
        cp = _convergent(confs)
        derivs = [cp.derivative(i) for i in range(len(confs))]
        assert all(d is not None for d in derivs)
        assert sum(derivs) == pytest.approx(confs[-1])

    @given(st.lists(
        st.floats(min_value=0.01, max_value=0.99),
        min_size=2, max_size=10,
    ))
    def test_limit_at_least_last_value_if_increasing(self, confs):
        """For a non-decreasing sequence, limit >= last value."""
        confs_sorted = sorted(confs)
        cp = _convergent(confs_sorted)
        lim = cp.limit
        if lim is not None:
            assert lim >= confs_sorted[-1] - 1e-9
