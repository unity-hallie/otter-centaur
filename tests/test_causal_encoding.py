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
    SymbolicEncoding,
    subdag,
    verify_gauge_computation,
    demo_linear_chain,
    demo_diamond,
    demo_spacetime_patch,
    build_nested_diamonds,
    cone_angle_limit,
    verify_cone_convergence,
    coordinate_free_gram,
    verify_frame_invariance,
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


# -- Property 7: Correlated path pairs -----------------------------------

class TestCorrelatedPaths:
    """
    Property 7: <v(X), v(Y)> = Σ_E paths(E→X) × paths(E→Y)
    The inner product counts correlated causal path pairs.
    """

    def test_property_7_diamond(self):
        """On the diamond DAG, verify <v(A),v(D)> = Σ paths(E→A)*paths(E→D)."""
        dag = _make_diamond()
        enc = CausalEncoding(dag)
        report = enc.correlated_path_pairs('A', 'D')
        ip = causal_hilbert_product(enc, 'A', 'D')
        assert report['correlated_pairs'] == ip

    @pytest.mark.parametrize("dag_factory", [
        _make_linear, _make_diamond, _make_spacetime,
    ])
    def test_property_7_all_pairs(self, dag_factory):
        """For every pair in every demo DAG, correlated_path_pairs == inner product."""
        dag = dag_factory()
        enc = CausalEncoding(dag)
        names = list(dag.events.keys())
        for x in names:
            for y in names:
                report = enc.correlated_path_pairs(x, y)
                ip = causal_hilbert_product(enc, x, y)
                assert report['correlated_pairs'] == ip, \
                    f"Property 7 failed for ({x},{y}): {report['correlated_pairs']} != {ip}"

    def test_from_root_equals_path_count(self):
        """When one argument is a root (single prime), ip = path count."""
        dag = _make_diamond()
        enc = CausalEncoding(dag)
        # A is a root: v(A) = {p_A: 1}, so <v(A), v(X)> = exp of p_A in gn(X) = paths(A→X)
        for name in dag.topological_order():
            ip = causal_hilbert_product(enc, 'A', name)
            pc = enc.path_count('A', name) if name != 'A' else 1
            assert ip == pc, f"<v(A),v({name})> = {ip} but paths(A→{name}) = {pc}"

    def test_bipartite_fork(self):
        """Property 7 on a wider DAG: A → {B,C,D} → E."""
        dag = CausalDAG()
        dag.add('A')
        dag.add('B', causes=['A'])
        dag.add('C', causes=['A'])
        dag.add('D', causes=['A'])
        dag.add('E', causes=['B', 'C', 'D'])
        enc = CausalEncoding(dag)
        # <v(A), v(E)> should equal 3 (3 paths from A to E)
        ip = causal_hilbert_product(enc, 'A', 'E')
        assert ip == 3
        report = enc.correlated_path_pairs('A', 'E')
        assert report['correlated_pairs'] == 3


# -- Property 8: The causal cone -----------------------------------------

class TestCausalCone:
    """
    Property 8: For k-ary nested diamonds, overlap(root, tip) → √((k-1)/k).
    The geometry converges to a fixed angle. Binary gives 45°.
    """

    def test_nested_diamond_norm_formula(self):
        """||v(Dn)||^2 = (2^{2n+1} - 1) for binary diamonds."""
        for n in range(1, 7):
            dag, tip = build_nested_diamonds(n, k=2)
            enc = CausalEncoding(dag)
            v = exponent_vector(enc, tip)
            norm_sq = sum(e ** 2 for e in v.values())
            predicted = (2 ** (2 * n + 1) - 1) // (2 - 1)
            assert norm_sq == predicted, \
                f"n={n}: ||v||^2={norm_sq} but expected {predicted}"

    def test_nested_diamond_overlap_converges(self):
        """Overlap converges to 1/sqrt(2) for binary diamonds."""
        limit = 1.0 / math.sqrt(2)
        dag, tip = build_nested_diamonds(8, k=2)
        enc = CausalEncoding(dag)
        ov = causal_overlap(enc, 'A', tip)
        assert ov == approx(limit, abs=1e-4)

    def test_cone_angle_limit_binary(self):
        """cone_angle_limit(2) gives 45 degrees."""
        cl = cone_angle_limit(2)
        assert cl['angle_degrees'] == approx(45.0, abs=0.01)
        assert cl['overlap'] == approx(1.0 / math.sqrt(2), abs=1e-10)

    def test_cone_angle_limit_general(self):
        """General k-ary limits match √((k-1)/k)."""
        for k in [3, 4, 5, 10]:
            cl = cone_angle_limit(k)
            expected = math.sqrt((k - 1) / k)
            assert cl['overlap'] == approx(expected, abs=1e-10), \
                f"k={k}: expected {expected}, got {cl['overlap']}"

    def test_kary_nested_diamonds_norm(self):
        """Norm formula works for k=3 (ternary diamonds)."""
        k = 3
        for n in range(1, 5):
            dag, tip = build_nested_diamonds(n, k=k)
            enc = CausalEncoding(dag)
            v = exponent_vector(enc, tip)
            norm_sq = sum(e ** 2 for e in v.values())
            predicted = (k ** (2 * n + 1) - 1) // (k - 1)
            assert norm_sq == predicted, \
                f"k={k}, n={n}: ||v||^2={norm_sq} but expected {predicted}"

    def test_verify_cone_convergence(self):
        """Full convergence verification via verify_cone_convergence."""
        report = verify_cone_convergence(max_n=6, k=2)
        assert report['all_norms_match']
        assert report['converges']
        assert report['limit_angle_degrees'] == approx(45.0, abs=0.01)

    def test_path_count_doubling(self):
        """Each diamond nesting doubles the path count for binary."""
        for n in range(1, 7):
            dag, tip = build_nested_diamonds(n, k=2)
            enc = CausalEncoding(dag)
            paths = enc.path_count('A', tip)
            assert paths == 2 ** n, f"n={n}: expected 2^{n}={2**n} paths, got {paths}"


# -- Property 9: Frame invariance -----------------------------------------

class TestFrameInvariance:
    """
    Property 9: The Gram matrix G_{XY} = Σ_E paths(E→X)·paths(E→Y)
    is computable from the DAG alone and is invariant under any prime assignment.
    """

    def test_coordinate_free_gram_matches_encoded(self):
        """The coordinate-free Gram matrix matches the one from any encoding."""
        dag = _make_diamond()
        enc = CausalEncoding(dag)
        names = dag.topological_order()

        G_free = coordinate_free_gram(dag)['matrix']
        G_enc = gram_matrix(enc)['matrix']

        for a in names:
            for b in names:
                assert G_free[(a, b)] == G_enc[(a, b)], \
                    f"Mismatch at ({a},{b}): free={G_free[(a,b)]}, enc={G_enc[(a,b)]}"

    @pytest.mark.parametrize("dag_factory", [
        _make_linear, _make_diamond, _make_spacetime,
    ])
    def test_frame_invariance(self, dag_factory):
        """Random prime assignments all produce the same Gram matrix."""
        dag = dag_factory()
        report = verify_frame_invariance(dag, num_trials=5)
        assert report['all_match'], \
            f"Frame invariance failed on {dag_factory.__name__}"

    def test_coordinate_free_gram_full_rank(self):
        """The Gram matrix has full rank (= number of events)."""
        for dag_factory in [_make_linear, _make_diamond, _make_spacetime]:
            dag = dag_factory()
            G = coordinate_free_gram(dag)
            assert G['rank'] == len(dag.events)

    def test_coordinate_free_gram_symmetric(self):
        """The coordinate-free Gram matrix is symmetric."""
        dag = _make_diamond()
        G = coordinate_free_gram(dag)['matrix']
        names = dag.topological_order()
        for a in names:
            for b in names:
                assert G[(a, b)] == G[(b, a)]

    def test_coordinate_free_gram_integer_entries(self):
        """All entries of the Gram matrix are non-negative integers."""
        dag = _make_diamond()
        G = coordinate_free_gram(dag)['matrix']
        for key, val in G.items():
            assert isinstance(val, int) and val >= 0, \
                f"G[{key}] = {val} is not a non-negative integer"

    def test_euler_factors_distinct_for_distinct_primes(self):
        """Gauge theorem part (C): w_p(t) != w_q(t) for distinct primes, generic t."""
        # This is proved by linear independence of cos(t ln p) and cos(t ln q)
        # when ln(p)/ln(q) is irrational (which it is for distinct primes, by FTA).
        # Here we verify numerically.
        from otter.causal_calculus import _first_n_primes
        test_primes = _first_n_primes(10)
        for i, p in enumerate(test_primes):
            for q in test_primes[i+1:]:
                for t_int in range(1, 50):
                    t = t_int * 0.1
                    s = complex(0.5, t)
                    wp = 1.0 / (1.0 - p ** (-s))
                    wq = 1.0 / (1.0 - q ** (-s))
                    assert abs(wp - wq) > 1e-10, \
                        f"w_{p}({t}) = w_{q}({t}): gauge group not free"

    def test_amplitudes_change_between_frames(self):
        """Gauge theorem: same DAG, different primes → different Born probabilities."""
        dag = _make_diamond()
        enc0 = CausalEncoding(dag)
        t = 3.0
        probs0 = born_probabilities(enc0, t)

        # Build a shifted encoding by constructing a new DAG with same structure
        # but CausalEncoding will assign different primes if we pad with dummies
        dag2 = CausalDAG()
        dag2.add("pad1")  # consume prime 2
        dag2.add("pad2")  # consume prime 3
        dag2.add("pad3")  # consume prime 5
        dag2.add("A")
        dag2.add("B", causes=["A"])
        dag2.add("C", causes=["A"])
        dag2.add("D", causes=["B", "C"])
        enc2 = CausalEncoding(dag2)
        probs2 = born_probabilities(enc2, t)

        # Probabilities should differ (different primes for same causal structure)
        diffs = [abs(probs0[n] - probs2[n]) for n in ['A', 'B', 'C', 'D']]
        assert max(diffs) > 0.01, \
            "Expected different Born probabilities in different frames"


# -- Property 10: Gauge computation ----------------------------------------

class TestGaugeComputation:
    """
    Property 10: The Gram matrix is computable without choosing primes,
    and independently encoded sub-DAGs produce consistent local Gram matrices.

    This is Property 9 used as a computational tool.
    """

    # --- Symbolic basics (P10a: deferred assignment) ---

    def test_symbolic_path_count_matches_concrete(self):
        """SymbolicEncoding path counts match CausalEncoding."""
        dag = _make_diamond()
        sym = SymbolicEncoding(dag)
        enc = CausalEncoding(dag)
        for anc in dag.events:
            for desc in dag.events:
                assert sym.path_count(anc, desc) == enc.path_count(anc, desc)

    def test_symbolic_gram_matches_coordinate_free(self):
        """SymbolicEncoding Gram matrix matches coordinate_free_gram."""
        dag = _make_diamond()
        sym = SymbolicEncoding(dag)
        G_sym = sym.gram_matrix()['matrix']
        G_free = coordinate_free_gram(dag)['matrix']
        for key in G_free:
            assert G_sym[key] == G_free[key]

    def test_symbolic_gram_matches_any_encoding(self):
        """SymbolicEncoding Gram matrix matches gram_matrix on spacetime DAG."""
        dag = _make_spacetime()
        sym = SymbolicEncoding(dag)
        enc = CausalEncoding(dag)
        G_sym = sym.gram_matrix()['matrix']
        G_enc = gram_matrix(enc)['matrix']
        names = dag.topological_order()
        for a in names:
            for b in names:
                assert G_sym[(a, b)] == G_enc[(a, b)]

    def test_symbolic_overlap_matches_concrete(self):
        """SymbolicEncoding overlaps match causal_overlap."""
        dag = _make_diamond()
        sym = SymbolicEncoding(dag)
        enc = CausalEncoding(dag)
        for a in dag.events:
            for b in dag.events:
                assert sym.overlap(a, b) == approx(causal_overlap(enc, a, b))

    def test_symbolic_norm_matches_concrete(self):
        """SymbolicEncoding norms match causal_hilbert_product."""
        dag = _make_diamond()
        sym = SymbolicEncoding(dag)
        enc = CausalEncoding(dag)
        for name in dag.events:
            assert sym.norm_sq(name) == causal_hilbert_product(enc, name, name)

    def test_symbolic_caching(self):
        """Path counts and Gram matrix are cached after first computation."""
        dag = _make_diamond()
        sym = SymbolicEncoding(dag)
        _ = sym.path_count('A', 'D')
        assert ('A', 'D') in sym._path_cache
        assert sym.path_count('A', 'D') == 2  # A->B->D and A->C->D

        G1 = sym.gram_matrix()
        G2 = sym.gram_matrix()
        assert G1 is G2  # Same object, cached

    # --- Materialization (P10a → concrete) ---

    def test_materialize_default(self):
        """Default materialization matches standard CausalEncoding."""
        dag = _make_diamond()
        sym = SymbolicEncoding(dag)
        enc_sym = sym.materialize()
        enc_std = CausalEncoding(dag)
        for name in dag.events:
            assert enc_sym.gn(name) == enc_std.gn(name)

    def test_materialize_custom_primes(self):
        """Custom primes produce a valid encoding with correct causality."""
        dag = _make_diamond()
        sym = SymbolicEncoding(dag)
        custom = {'A': 7, 'B': 11, 'C': 13, 'D': 17}
        enc = sym.materialize(primes=custom)
        # Causality preserved
        assert enc.causes('A', 'B')
        assert enc.causes('A', 'C')
        assert enc.causes('A', 'D')
        assert enc.causes('B', 'D')
        assert enc.causes('C', 'D')
        assert not enc.causes('B', 'A')
        assert not enc.causes('B', 'C')

    def test_materialize_custom_gram_invariant(self):
        """Custom primes produce the same Gram matrix (Property 9)."""
        dag = _make_diamond()
        sym = SymbolicEncoding(dag)
        G_sym = sym.gram_matrix()['matrix']

        custom = {'A': 7, 'B': 11, 'C': 13, 'D': 17}
        enc = sym.materialize(primes=custom)
        G_enc = gram_matrix(enc)['matrix']

        for key in G_sym:
            assert G_sym[key] == G_enc[key]

    # --- Sub-DAG decomposition (P10b: local encoding) ---

    def test_subdag_preserves_structure(self):
        """subdag extracts correct events and causes."""
        dag = _make_diamond()
        sub = subdag(dag, {'A', 'B', 'D'})
        assert 'A' in sub.events
        assert 'B' in sub.events
        assert 'D' in sub.events
        assert 'C' not in sub.events
        assert sub.events['B'].causes == frozenset(['A'])
        # D's cause C was excluded, only B remains
        assert sub.events['D'].causes == frozenset(['B'])

    def test_subdag_root_promotion(self):
        """Events whose causes are outside the subset become roots."""
        dag = _make_diamond()
        sub = subdag(dag, {'B', 'C', 'D'})
        # B and C had cause A, but A is not in subset -> they become roots
        assert sub.events['B'].causes == frozenset()
        assert sub.events['C'].causes == frozenset()

    def test_subdag_path_counts_are_local(self):
        """
        The honesty test: sub-DAG path counts differ from full DAG
        when paths through excluded events are lost.

        In the full diamond, paths(A, D) = 2 (through B and through C).
        In sub-DAG {A, B, D}, paths(A, D) = 1 (only through B).

        This is not a bug. It is the thing to be honest about.
        Local geometry is correct within each piece.
        Full geometry requires the full structure.
        Neither requires specific primes.
        """
        dag = _make_diamond()

        # Full DAG: 2 paths from A to D
        sym_full = SymbolicEncoding(dag)
        assert sym_full.path_count('A', 'D') == 2

        # Sub-DAG {A, B, D}: only 1 path from A to D
        sub = subdag(dag, {'A', 'B', 'D'})
        sym_sub = SymbolicEncoding(sub)
        assert sym_sub.path_count('A', 'D') == 1

        # Therefore G(A,D) differs between full and sub
        G_full = sym_full.gram_matrix()['matrix']
        G_sub = sym_sub.gram_matrix()['matrix']
        assert G_full[('A', 'D')] != G_sub[('A', 'D')]

        # But roots have the same self-norm in both
        # (A is a root in both, with no ancestors)
        assert G_full[('A', 'A')] == G_sub[('A', 'A')]

    # --- Independent encoding (P10b+c: merge) ---

    def test_independent_encodings_local_gram_correct(self):
        """Random primes on a sub-DAG produce the correct local Gram matrix."""
        dag = _make_spacetime()
        sub = subdag(dag, {'past_1', 'past_2', 'now'})
        report = verify_gauge_computation(dag, [{'past_1', 'past_2', 'now'}])
        assert report['all_match']

    def test_parallel_workers_different_primes(self):
        """
        Two workers encode overlapping sub-DAGs with completely different
        primes. Both produce correct local Gram matrices.

        This is the test that proves P10 is not just P9 restated:
        it is P9 used as a computational strategy.
        """
        dag = _make_spacetime()

        # Worker 1: {past_1, past_2, now} with primes {2, 3, 5}
        sub1 = subdag(dag, {'past_1', 'past_2', 'now'})
        sym1 = SymbolicEncoding(sub1)
        enc1 = sym1.materialize(primes={'past_1': 2, 'past_2': 3, 'now': 5})

        # Worker 2: {past_2, other, now, future} with primes {7, 11, 13, 17}
        sub2 = subdag(dag, {'past_2', 'other', 'now', 'future'})
        sym2 = SymbolicEncoding(sub2)
        enc2 = sym2.materialize(primes={
            'past_2': 7, 'other': 11, 'now': 13, 'future': 17,
        })

        # Both workers' Gram matrices match their coordinate-free references
        G1 = gram_matrix(enc1)['matrix']
        G1_ref = coordinate_free_gram(sub1)['matrix']
        for a in sub1.events:
            for b in sub1.events:
                assert G1[(a, b)] == G1_ref[(a, b)]

        G2 = gram_matrix(enc2)['matrix']
        G2_ref = coordinate_free_gram(sub2)['matrix']
        for a in sub2.events:
            for b in sub2.events:
                assert G2[(a, b)] == G2_ref[(a, b)]

    @pytest.mark.parametrize("dag_factory", [
        _make_linear, _make_diamond, _make_spacetime,
    ])
    def test_verify_gauge_computation_parametrized(self, dag_factory):
        """Gauge computation verified across different DAG topologies."""
        dag = dag_factory()
        names = list(dag.events.keys())
        # Partition into individual events (trivial but valid)
        partition = [{n} for n in names]
        report = verify_gauge_computation(dag, partition)
        assert report['all_match']

    def test_verify_gauge_computation_nontrivial(self):
        """Non-trivial overlapping partition of spacetime DAG."""
        dag = _make_spacetime()
        report = verify_gauge_computation(dag, [
            {'past_1', 'past_2', 'now'},
            {'past_2', 'other', 'now', 'future'},
        ])
        assert report['all_match']
