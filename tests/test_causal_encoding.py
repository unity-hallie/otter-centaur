"""
Unit tests for the causal encoding domain.

Core claims:
    - CausalDAG correctly computes topological order, roots, children, ancestors
    - CausalEncoding assigns distinct fresh primes and injective Godel numbers
    - Divisibility tracks causality (Claim 2) on all demo DAGs
    - GCD = common ancestors, LCM = causal join
    - Exponent vectors correctly represent prime factorizations
    - The causal Hilbert product encodes orthogonality = causal independence
    - Causal overlap ranges from 0 (independent) to 1 (self)
    - Gram matrix is symmetric, positive semidefinite, correct dimension
    - Gleason's theorem applies for dim >= 3
    - Path amplitudes: single-path events have 0 interference
    - Path amplitudes: multi-path events (diamond) have nonzero interference
    - Single-prime interference theorem holds (constructive always)
    - Born probabilities sum to 1.0 at all t values
    - Deep fork DAGs exhibit destructive interference
"""

import pytest
import math
from pytest import approx

from otter.domains.causal_encoding import (
    CausalDAG,
    CausalEncoding,
    demo_linear_chain,
    demo_diamond,
    demo_spacetime_patch,
    causal_amplitude,
    born_probabilities,
    exponent_vector,
    causal_hilbert_product,
    causal_norm,
    causal_overlap,
    gram_matrix,
    path_amplitudes,
    single_prime_interference_theorem,
)


# -- Helpers -----------------------------------------------------------------

def _make_linear():
    """A -> B -> C -> D"""
    return demo_linear_chain()


def _make_diamond():
    """A -> B, A -> C, B -> D, C -> D"""
    return demo_diamond()


def _make_spacetime():
    """past_1, past_2 -> now, past_2 -> other, now + other -> future"""
    return demo_spacetime_patch()


def _make_deep_fork():
    """A -> B -> C -> F, A -> D -> E -> F (2 unique primes per path)"""
    dag = CausalDAG()
    dag.add('A')
    dag.add('B', causes=['A'])
    dag.add('C', causes=['B'])
    dag.add('D', causes=['A'])
    dag.add('E', causes=['D'])
    dag.add('F', causes=['C', 'E'])
    return dag


# == 1. CausalDAG basics ====================================================

class TestCausalDAGBasics:
    def test_topological_order_linear(self):
        dag = _make_linear()
        order = dag.topological_order()
        assert order == ["A", "B", "C", "D"]

    def test_topological_order_diamond(self):
        dag = _make_diamond()
        order = dag.topological_order()
        # A must come first, D must come last
        assert order[0] == "A"
        assert order[-1] == "D"
        # B and C between A and D
        assert set(order[1:3]) == {"B", "C"}

    def test_roots_linear(self):
        dag = _make_linear()
        roots = dag.roots()
        assert len(roots) == 1
        assert roots[0].name == "A"

    def test_roots_spacetime(self):
        dag = _make_spacetime()
        root_names = {e.name for e in dag.roots()}
        assert root_names == {"past_1", "past_2"}

    def test_children(self):
        dag = _make_diamond()
        children_of_a = dag.children("A")
        child_names = {e.name for e in children_of_a}
        assert child_names == {"B", "C"}

    def test_children_leaf(self):
        dag = _make_diamond()
        assert dag.children("D") == []

    def test_ancestors_linear(self):
        dag = _make_linear()
        assert dag.ancestors("D") == {"A", "B", "C"}
        assert dag.ancestors("B") == {"A"}
        assert dag.ancestors("A") == set()

    def test_ancestors_diamond(self):
        dag = _make_diamond()
        assert dag.ancestors("D") == {"A", "B", "C"}
        assert dag.ancestors("B") == {"A"}
        assert dag.ancestors("C") == {"A"}


# == 2. CausalEncoding: primes and injectivity ==============================

class TestCausalEncodingBasics:
    def test_fresh_primes_distinct(self):
        dag = _make_diamond()
        enc = CausalEncoding(dag)
        primes = [enc.fresh_prime(n) for n in dag.events]
        assert len(set(primes)) == len(primes), "Fresh primes must be distinct"

    def test_gn_injective(self):
        dag = _make_diamond()
        enc = CausalEncoding(dag)
        gns = [enc.gn(n) for n in dag.events]
        assert len(set(gns)) == len(gns), "Godel numbers must be injective"

    def test_divisibility_equals_causality_linear(self):
        dag = _make_linear()
        enc = CausalEncoding(dag)
        # A causes B, C, D
        assert enc.causes("A", "B")
        assert enc.causes("A", "C")
        assert enc.causes("A", "D")
        # B does not cause A
        assert not enc.causes("B", "A")
        # C does not cause B
        assert not enc.causes("C", "B")

    def test_root_gn_is_its_fresh_prime(self):
        dag = _make_linear()
        enc = CausalEncoding(dag)
        # Root event with no causes should have gn = fresh_prime
        assert enc.gn("A") == enc.fresh_prime("A")


# == 3. verify_claim_2 on all demo DAGs =====================================

class TestVerifyClaim2:
    @pytest.mark.parametrize("dag_factory", [
        _make_linear, _make_diamond, _make_spacetime,
    ])
    def test_claim_2_holds(self, dag_factory):
        dag = dag_factory()
        enc = CausalEncoding(dag)
        report = enc.verify_claim_2()
        assert report["claim_2_holds"] is True
        assert report["counterexamples"] == []
        assert report["verified"] == report["total_pairs"]


# == 4. GCD = common ancestors (diamond) ====================================

class TestGCDCommonAncestors:
    def test_diamond_b_c_share_only_a(self):
        dag = _make_diamond()
        enc = CausalEncoding(dag)
        common = enc.common_ancestors("B", "C")
        # B and C share only ancestor A (plus themselves are counted
        # only if their prime divides the gcd, which it won't since
        # B's prime doesn't divide C's gn and vice versa)
        assert "A" in common
        assert "B" not in common
        assert "C" not in common

    def test_diamond_a_d_ancestor_set(self):
        dag = _make_diamond()
        enc = CausalEncoding(dag)
        common = enc.common_ancestors("A", "D")
        # A's gn divides D's gn, so A is a common ancestor
        assert "A" in common


# == 5. LCM = causal join (diamond) =========================================

class TestLCMCausalJoin:
    def test_lcm_contains_both_ancestries(self):
        dag = _make_diamond()
        enc = CausalEncoding(dag)
        lcm_val = enc.causal_join_number("B", "C")
        # LCM must be divisible by both gn(B) and gn(C)
        assert lcm_val % enc.gn("B") == 0
        assert lcm_val % enc.gn("C") == 0

    def test_lcm_symmetric(self):
        dag = _make_diamond()
        enc = CausalEncoding(dag)
        assert enc.causal_join_number("B", "C") == enc.causal_join_number("C", "B")

    def test_lcm_self_is_self(self):
        dag = _make_diamond()
        enc = CausalEncoding(dag)
        assert enc.causal_join_number("A", "A") == enc.gn("A")


# == 6. exponent_vector ======================================================

class TestExponentVector:
    def test_root_event_single_prime(self):
        dag = _make_linear()
        enc = CausalEncoding(dag)
        vec = exponent_vector(enc, "A")
        # Root has gn = fresh_prime, so single entry with exponent 1
        assert len(vec) == 1
        assert list(vec.values()) == [1]

    def test_linear_chain_accumulates_primes(self):
        dag = _make_linear()
        enc = CausalEncoding(dag)
        # Each step adds one more prime
        vec_a = exponent_vector(enc, "A")
        vec_b = exponent_vector(enc, "B")
        vec_c = exponent_vector(enc, "C")
        vec_d = exponent_vector(enc, "D")
        assert len(vec_a) == 1
        assert len(vec_b) == 2
        assert len(vec_c) == 3
        assert len(vec_d) == 4

    def test_diamond_d_has_four_primes(self):
        dag = _make_diamond()
        enc = CausalEncoding(dag)
        vec_d = exponent_vector(enc, "D")
        # D's gn = p_D * gn(B) * gn(C) = p_D * (p_B * p_A) * (p_C * p_A)
        # Primes: p_A (exponent 2), p_B, p_C, p_D
        assert len(vec_d) == 4
        # A's prime should have exponent 2 (inherited through both B and C)
        p_a = enc.fresh_prime("A")
        assert vec_d[p_a] == 2

    def test_all_exponents_positive(self):
        dag = _make_spacetime()
        enc = CausalEncoding(dag)
        for name in dag.events:
            vec = exponent_vector(enc, name)
            for exp in vec.values():
                assert exp > 0


# == 7. causal_hilbert_product ===============================================

class TestCausalHilbertProduct:
    def test_orthogonal_for_independent_events(self):
        dag = _make_diamond()
        enc = CausalEncoding(dag)
        # B and C share ancestor A, so they are NOT independent
        # Need a DAG with truly independent events
        dag2 = CausalDAG()
        dag2.add("X")
        dag2.add("Y")
        enc2 = CausalEncoding(dag2)
        assert causal_hilbert_product(enc2, "X", "Y") == 0

    def test_nonzero_for_related_events(self):
        dag = _make_diamond()
        enc = CausalEncoding(dag)
        # A causes B, so they share A's prime
        assert causal_hilbert_product(enc, "A", "B") > 0

    def test_self_product_equals_prime_count_squared(self):
        dag = _make_linear()
        enc = CausalEncoding(dag)
        # For the causal encoding where exponents are all 1 along a chain,
        # <v(B), v(B)> = sum of a_p^2 = number of primes (each with exponent 1)
        # B has primes p_A, p_B -> product = 1*1 + 1*1 = 2
        assert causal_hilbert_product(enc, "B", "B") == 2

    def test_symmetric(self):
        dag = _make_diamond()
        enc = CausalEncoding(dag)
        assert causal_hilbert_product(enc, "A", "D") == \
               causal_hilbert_product(enc, "D", "A")


# == 8. causal_overlap =======================================================

class TestCausalOverlap:
    def test_independent_events_zero(self):
        dag = CausalDAG()
        dag.add("X")
        dag.add("Y")
        enc = CausalEncoding(dag)
        assert causal_overlap(enc, "X", "Y") == approx(0.0)

    def test_self_overlap_one(self):
        dag = _make_diamond()
        enc = CausalEncoding(dag)
        assert causal_overlap(enc, "A", "A") == approx(1.0)
        assert causal_overlap(enc, "D", "D") == approx(1.0)

    def test_partial_overlap_between_zero_and_one(self):
        dag = _make_diamond()
        enc = CausalEncoding(dag)
        overlap_bc = causal_overlap(enc, "B", "C")
        assert 0.0 < overlap_bc < 1.0

    def test_ancestor_descendant_relationship(self):
        dag = _make_linear()
        enc = CausalEncoding(dag)
        # A is contained within B's factorization, so overlap should be
        # high but depends on norms: overlap = 1 / sqrt(2) for A-B
        overlap_ab = causal_overlap(enc, "A", "B")
        assert 0.0 < overlap_ab <= 1.0


# == 9. gram_matrix: symmetric, positive semidefinite, correct dim ===========

class TestGramMatrix:
    def test_symmetric(self):
        dag = _make_diamond()
        enc = CausalEncoding(dag)
        gm = gram_matrix(enc)
        names = gm['names']
        for a in names:
            for b in names:
                assert gm['matrix'][(a, b)] == gm['matrix'][(b, a)]

    def test_diagonal_positive(self):
        dag = _make_diamond()
        enc = CausalEncoding(dag)
        gm = gram_matrix(enc)
        for name in gm['names']:
            assert gm['matrix'][(name, name)] > 0

    def test_correct_dimension(self):
        dag = _make_diamond()
        enc = CausalEncoding(dag)
        gm = gram_matrix(enc)
        # Diamond has 4 events, each with a distinct fresh prime -> dim = 4
        assert gm['dim'] == 4
        assert gm['rank'] == 4

    def test_matrix_size(self):
        dag = _make_diamond()
        enc = CausalEncoding(dag)
        gm = gram_matrix(enc)
        n = len(gm['names'])
        assert len(gm['matrix']) == n * n


# == 10. gram_matrix: Gleason applies for dim >= 3 ==========================

class TestGleasonApplies:
    def test_diamond_gleason_true(self):
        dag = _make_diamond()
        enc = CausalEncoding(dag)
        gm = gram_matrix(enc)
        assert gm['dim'] >= 3
        assert gm['gleason_applies'] is True

    def test_two_event_dag_gleason_false(self):
        dag = CausalDAG()
        dag.add("X")
        dag.add("Y", causes=["X"])
        enc = CausalEncoding(dag)
        gm = gram_matrix(enc)
        assert gm['dim'] == 2
        assert gm['gleason_applies'] is False

    def test_spacetime_gleason_true(self):
        dag = _make_spacetime()
        enc = CausalEncoding(dag)
        gm = gram_matrix(enc)
        assert gm['dim'] >= 3
        assert gm['gleason_applies'] is True


# == 11. path_amplitudes: single-path event has 0 interference ==============

class TestPathAmplitudesSinglePath:
    def test_root_event_no_interference(self):
        dag = _make_diamond()
        enc = CausalEncoding(dag)
        result = path_amplitudes(enc, "A", t=1.0)
        assert result['interference'] == approx(0.0)

    def test_linear_chain_no_interference(self):
        dag = _make_linear()
        enc = CausalEncoding(dag)
        # D has only one path from root: A->B->C->D
        result = path_amplitudes(enc, "D", t=5.0)
        assert result['interference'] == approx(0.0)


# == 12. path_amplitudes: multi-path event (diamond D) has nonzero interf ===

class TestPathAmplitudesMultiPath:
    def test_diamond_d_has_nonzero_interference(self):
        dag = _make_diamond()
        enc = CausalEncoding(dag)
        result = path_amplitudes(enc, "D", t=1.0)
        assert abs(result['interference']) > 1e-12

    def test_diamond_d_has_two_paths(self):
        dag = _make_diamond()
        enc = CausalEncoding(dag)
        result = path_amplitudes(enc, "D", t=1.0)
        assert len(result['paths']) == 2

    def test_diamond_d_shared_primes_include_root(self):
        dag = _make_diamond()
        enc = CausalEncoding(dag)
        result = path_amplitudes(enc, "D", t=1.0)
        p_a = enc.fresh_prime("A")
        p_d = enc.fresh_prime("D")
        assert p_a in result['shared_primes']
        assert p_d in result['shared_primes']


# == 13. single_prime_interference_theorem ===================================

class TestSinglePrimeInterferenceTheorem:
    def test_theorem_holds(self):
        assert single_prime_interference_theorem(verbose=False) is True


# == 14. born_probabilities: sums to 1.0 at multiple t values ===============

class TestBornProbabilities:
    @pytest.mark.parametrize("t", [0.0, 1.0, 3.0, 7.0, 10.0, 21.0])
    def test_sums_to_one(self, t):
        dag = _make_diamond()
        enc = CausalEncoding(dag)
        probs = born_probabilities(enc, t)
        assert sum(probs.values()) == approx(1.0, abs=1e-10)

    def test_all_probabilities_positive(self):
        dag = _make_diamond()
        enc = CausalEncoding(dag)
        probs = born_probabilities(enc, t=5.0)
        for p in probs.values():
            assert p > 0.0


# == 15. born_probabilities near Riemann zero (t~14) =========================

class TestBornProbabilitiesNearZero:
    def test_sums_to_one_near_riemann_zero(self):
        dag = _make_diamond()
        enc = CausalEncoding(dag)
        # t ~ 14.13 is near the first nontrivial Riemann zero
        probs = born_probabilities(enc, t=14.0)
        assert sum(probs.values()) == approx(1.0, abs=1e-10)

    def test_sums_to_one_at_14_13(self):
        dag = _make_diamond()
        enc = CausalEncoding(dag)
        probs = born_probabilities(enc, t=14.13)
        assert sum(probs.values()) == approx(1.0, abs=1e-10)


# == 16. Deep fork DAG: destructive interference exists ======================

class TestDeepForkDestructiveInterference:
    def test_destructive_interference_exists(self):
        dag = _make_deep_fork()
        enc = CausalEncoding(dag)
        # Sweep a range of t values to find destructive interference
        found_destructive = False
        for t_int in range(0, 120):
            t = float(t_int) * 0.25
            result = path_amplitudes(enc, "F", t)
            if result['interference'] < -1e-12:
                found_destructive = True
                break
        assert found_destructive, \
            "Deep fork (2+ unique primes per path) should exhibit destructive interference"

    def test_deep_fork_has_multiple_paths(self):
        dag = _make_deep_fork()
        enc = CausalEncoding(dag)
        result = path_amplitudes(enc, "F", t=1.0)
        assert len(result['paths']) == 2

    def test_deep_fork_path_unique_primes_count(self):
        dag = _make_deep_fork()
        enc = CausalEncoding(dag)
        result = path_amplitudes(enc, "F", t=1.0)
        # Each path should have 2 unique primes (B,C or D,E)
        for _, unique_primes, _ in result['path_unique']:
            assert len(unique_primes) == 2


# -- The Easter Egg: exponents count paths ----------------------------------

class TestPathCounting:
    """
    Property 6: the exponent of p_A in gn(E) equals the number of
    directed paths from A to E.

    Nobody asked for this. The encoding was designed so that divisibility
    tracks causality. But the exponents count paths for free.
    """

    def test_diamond_two_paths(self):
        """A→B→D and A→C→D: exponent of p_A in gn(D) should be 2."""
        dag = _make_diamond()
        enc = CausalEncoding(dag)
        p_A = enc.fresh_prime("A")
        gn_D = enc.gn("D")
        exp = 0
        n = gn_D
        while n % p_A == 0:
            n //= p_A
            exp += 1
        assert exp == 2

    def test_triple_merge_three_paths(self):
        """A→B, A→C, A→D, all→E: 3 paths from A to E."""
        dag = CausalDAG()
        dag.add('A')
        dag.add('B', causes=['A'])
        dag.add('C', causes=['A'])
        dag.add('D', causes=['A'])
        dag.add('E', causes=['B', 'C', 'D'])
        enc = CausalEncoding(dag)
        p_A = enc.fresh_prime("A")
        gn_E = enc.gn("E")
        exp = 0
        n = gn_E
        while n % p_A == 0:
            n //= p_A
            exp += 1
        assert exp == 3

    def test_double_diamond_four_paths(self):
        """Double diamond: 4 paths from A to F."""
        dag = CausalDAG()
        dag.add('A')
        dag.add('B', causes=['A'])
        dag.add('C', causes=['A'])
        dag.add('D', causes=['B', 'C'])
        dag.add('E', causes=['B', 'C'])
        dag.add('F', causes=['D', 'E'])
        enc = CausalEncoding(dag)
        p_A = enc.fresh_prime("A")
        gn_F = enc.gn("F")
        exp = 0
        n = gn_F
        while n % p_A == 0:
            n //= p_A
            exp += 1
        assert exp == 4

    def test_linear_chain_always_one_path(self):
        """In a linear chain, every ancestor has exactly 1 path to every descendant."""
        dag = _make_linear()
        enc = CausalEncoding(dag)
        names = dag.topological_order()
        for i, anc in enumerate(names):
            for desc in names[i+1:]:
                p = enc.fresh_prime(anc)
                n = enc.gn(desc)
                exp = 0
                while n % p == 0:
                    n //= p
                    exp += 1
                assert exp == 1, f"Expected 1 path {anc}→{desc}, got exponent {exp}"

    def test_verify_path_counting_all_dags(self):
        """Run the full verification on every demo DAG."""
        for maker in [_make_linear, _make_diamond, _make_spacetime]:
            dag = maker()
            enc = CausalEncoding(dag)
            result = enc.verify_path_counting()
            assert result["path_counting_holds"], \
                f"Path counting failed on {maker.__name__}: {result['counterexamples']}"
