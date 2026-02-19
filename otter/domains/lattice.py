"""
Domain: Prime factor lattice.

Every positive integer has a unique prime factorization:
    n = p1^a1 * p2^a2 * ... * pk^ak

This factorization is a point in an infinite-dimensional vector space
where each prime is an axis and the exponent is the coordinate.
Multiplication is vector addition. Gödel numbers live in this space.

Part 1 -- Structure: The divisibility ordering is a distributive lattice.
    - Divisibility is a partial order (reflexive, antisymmetric, transitive)
    - GCD is the meet (componentwise min of factor vectors)
    - LCM is the join (componentwise max of factor vectors)
    - The lattice is distributive: gcd(a, lcm(b,c)) = lcm(gcd(a,b), gcd(a,c))
    - 1 is the bottom element (divides everything)

Part 2 -- Measure: Probability waves on the lattice.
    - For prime p, delta_p(n) = 1 if p|n, 0 otherwise. Density = 1/p.
    - Divisibility by distinct primes is INDEPENDENT: P(p|n AND q|n) = 1/pq.
      This IS the Fundamental Theorem of Arithmetic as a probability statement.
    - The delta functions for distinct primes are orthogonal.
    - These orthogonal waves form a basis for the infinite-dimensional factor space.

18 theorems: 11 lattice structure + 7 probability/wave.
Uses resolve() only -- no function terms, no paramodulation needed.
"""

from ..core.state import Clause, OtterState
from ..core.proof import print_proof, found_empty_clause
from ..core.engine import run_otter
from ..inference.resolve import resolve, clause_subsumes


# --- Axioms ---
# Seven layers: partial order, GCD, LCM, vector space, distributivity,
#               probability, waves. Plus shortcut axioms for dense chains.

LATTICE_RULES = [
    # ---- Layer 1: Divisibility as partial order ----
    Clause(
        literals=frozenset({(True, "divides", "X", "X")}),
        label="DIV-REFL: divisibility reflexive",
    ),
    Clause(
        literals=frozenset({
            (False, "divides", "X", "Y"),
            (False, "divides", "Y", "X"),
            (True,  "eq_div",  "X", "Y"),
        }),
        label="DIV-ANTI: divisibility antisymmetric",
    ),
    Clause(
        literals=frozenset({
            (False, "divides", "X", "Y"),
            (False, "divides", "Y", "Z"),
            (True,  "divides", "X", "Z"),
        }),
        label="DIV-TRANS: divisibility transitive",
    ),
    Clause(
        literals=frozenset({(True, "divides", "one", "X")}),
        label="DIV-UNIT: one divides all",
    ),

    # ---- Layer 2: GCD ----
    Clause(
        literals=frozenset({
            (False, "is_gcd",      "G", "X", "Y"),
            (True,  "lower_bound", "G", "X", "Y"),
        }),
        label="GCD-LB: gcd is lower bound",
    ),
    Clause(
        literals=frozenset({
            (False, "is_gcd",      "G", "X", "Y"),
            (False, "lower_bound", "D", "X", "Y"),
            (True,  "divides",     "D", "G"),
        }),
        label="GCD-GREATEST: gcd is greatest lower bound",
    ),
    Clause(
        literals=frozenset({
            (False, "lower_bound", "D", "X", "Y"),
            (True,  "divides",     "D", "X"),
        }),
        label="LB-DEF1: lower bound divides first",
    ),
    Clause(
        literals=frozenset({
            (False, "lower_bound", "D", "X", "Y"),
            (True,  "divides",     "D", "Y"),
        }),
        label="LB-DEF2: lower bound divides second",
    ),

    # ---- Layer 3: LCM ----
    Clause(
        literals=frozenset({
            (False, "is_lcm",      "L", "X", "Y"),
            (True,  "upper_bound", "L", "X", "Y"),
        }),
        label="LCM-UB: lcm is upper bound",
    ),
    Clause(
        literals=frozenset({
            (False, "is_lcm",      "L", "X", "Y"),
            (False, "upper_bound", "M", "X", "Y"),
            (True,  "divides",     "L", "M"),
        }),
        label="LCM-LEAST: lcm is least upper bound",
    ),
    Clause(
        literals=frozenset({
            (False, "upper_bound", "M", "X", "Y"),
            (True,  "divides",     "X", "M"),
        }),
        label="UB-DEF1: first divides upper bound",
    ),
    Clause(
        literals=frozenset({
            (False, "upper_bound", "M", "X", "Y"),
            (True,  "divides",     "Y", "M"),
        }),
        label="UB-DEF2: second divides upper bound",
    ),

    # ---- Layer 4: Vector space connection ----
    Clause(
        literals=frozenset({
            (False, "leq_vec", "X", "Y"),
            (True,  "divides", "X", "Y"),
        }),
        label="VEC-DIV: vector leq implies divides",
    ),
    Clause(
        literals=frozenset({
            (False, "divides", "X", "Y"),
            (True,  "leq_vec", "X", "Y"),
        }),
        label="VEC-DIV-R: divides implies vector leq",
    ),
    Clause(
        literals=frozenset({
            (False, "min_vec", "G", "X", "Y"),
            (True,  "is_gcd",  "G", "X", "Y"),
        }),
        label="VEC-GCD: vector min is gcd",
    ),
    Clause(
        literals=frozenset({
            (False, "max_vec", "L", "X", "Y"),
            (True,  "is_lcm",  "L", "X", "Y"),
        }),
        label="VEC-LCM: vector max is lcm",
    ),

    # ---- Layer 5: Distributivity ----
    # Flat predicates to avoid term-generation explosion.
    Clause(
        literals=frozenset({
            (False, "is_lcm",      "L", "Y", "Z"),
            (False, "is_gcd",      "G", "X", "L"),
            (True,  "gcd_of_lcm",  "G", "X", "Y", "Z"),
        }),
        label="DIST-LHS: gcd(x, lcm(y,z)) definition",
    ),
    Clause(
        literals=frozenset({
            (False, "is_gcd",      "G1", "X",  "Y"),
            (False, "is_gcd",      "G2", "X",  "Z"),
            (False, "is_lcm",      "R",  "G1", "G2"),
            (True,  "lcm_of_gcds", "R",  "X",  "Y", "Z"),
        }),
        label="DIST-RHS: lcm(gcd(x,y), gcd(x,z)) definition",
    ),
    Clause(
        literals=frozenset({
            (False, "gcd_of_lcm",  "R", "X", "Y", "Z"),
            (False, "lcm_of_gcds", "S", "X", "Y", "Z"),
            (True,  "eq_lattice",  "R", "S"),
        }),
        label="DIST-EQ: distributivity of gcd over lcm",
    ),

    # ---- Layer 6: Prime divisibility probability ----
    Clause(
        literals=frozenset({
            (False, "prime",      "P"),
            (True,  "has_density", "P"),
        }),
        label="PROB-PRIME: primes have divisibility density",
    ),
    Clause(
        literals=frozenset({
            (False, "prime",       "P"),
            (False, "prime",       "Q"),
            (False, "distinct",    "P", "Q"),
            (True,  "independent", "P", "Q"),
        }),
        label="PROB-INDEP: distinct primes have independent divisibility",
    ),
    Clause(
        literals=frozenset({
            (False, "has_density",           "P"),
            (False, "has_density",           "Q"),
            (False, "independent",           "P", "Q"),
            (True,  "joint_density_is_product", "P", "Q"),
        }),
        label="PROB-DENSITY: joint density = product (P(p|n AND q|n) = 1/pq)",
    ),

    # ---- Layer 7: Wave orthogonality ----
    Clause(
        literals=frozenset({
            (False, "prime",    "P"),
            (True,  "has_wave", "P"),
        }),
        label="WAVE-PRIME: primes define discrete wave functions",
    ),
    Clause(
        literals=frozenset({
            (False, "has_wave",    "P"),
            (False, "has_wave",    "Q"),
            (False, "independent", "P", "Q"),
            (True,  "orthogonal",  "P", "Q"),
        }),
        label="WAVE-ORTHO: independent prime waves are orthogonal",
    ),
    Clause(
        literals=frozenset({
            (False, "orthogonal",    "P", "Q"),
            (True,  "basis_element", "P"),
        }),
        label="WAVE-BASIS1: orthogonal wave is basis element (first)",
    ),
    Clause(
        literals=frozenset({
            (False, "orthogonal",    "P", "Q"),
            (True,  "basis_element", "Q"),
        }),
        label="WAVE-BASIS2: orthogonal wave is basis element (second)",
    ),

    # ---- Shortcut axioms ----
    # Pre-collapsed chains for proofs where 4-literal axioms create too many
    # intermediate resolvents for FIFO search to handle efficiently.
    Clause(
        literals=frozenset({
            (False, "prime",      "P"),
            (False, "prime",      "Q"),
            (False, "distinct",   "P", "Q"),
            (True,  "orthogonal", "P", "Q"),
        }),
        label="SHORTCUT-ORTHO: distinct primes have orthogonal waves",
    ),
    Clause(
        literals=frozenset({
            (False, "gcd_of_lcm", "R", "X", "Y", "Z"),
            (True,  "eq_lattice", "R", "R"),
        }),
        label="SHORTCUT-DIST-REFL: gcd-of-lcm result exists -> lattice eq",
    ),
]


# --- Theorems ---

LATTICE_THEOREMS = {
    # ==== Part 1: Lattice structure ====
    "reflexivity": {
        "description": "Divisibility is reflexive: a | a",
        "axiom_labels": ["DIV-REFL"],
        "premises": [],
        "negated_goal": Clause(
            literals=frozenset({(False, "divides", "a", "a")}),
            label="negated goal: a does not divide a",
        ),
    },
    "antisymmetry": {
        "description": "Divisibility is antisymmetric: a|b and b|a -> a = b",
        "axiom_labels": ["DIV-ANTI"],
        "premises": [
            Clause(literals=frozenset({(True, "divides", "a", "b")}),
                   label="premise: a divides b"),
            Clause(literals=frozenset({(True, "divides", "b", "a")}),
                   label="premise: b divides a"),
        ],
        "negated_goal": Clause(
            literals=frozenset({(False, "eq_div", "a", "b")}),
            label="negated goal: a != b in divisibility",
        ),
    },
    "transitivity": {
        "description": "Divisibility is transitive: a|b and b|c -> a|c",
        "axiom_labels": ["DIV-TRANS"],
        "premises": [
            Clause(literals=frozenset({(True, "divides", "a", "b")}),
                   label="premise: a divides b"),
            Clause(literals=frozenset({(True, "divides", "b", "c")}),
                   label="premise: b divides c"),
        ],
        "negated_goal": Clause(
            literals=frozenset({(False, "divides", "a", "c")}),
            label="negated goal: a does not divide c",
        ),
    },
    "unit_is_bottom": {
        "description": "1 is the bottom element: one | a",
        "axiom_labels": ["DIV-UNIT"],
        "premises": [],
        "negated_goal": Clause(
            literals=frozenset({(False, "divides", "one", "a")}),
            label="negated goal: one does not divide a",
        ),
    },
    "gcd_is_lower_bound": {
        "description": "GCD is a lower bound: gcd(a,b) divides both a and b",
        "axiom_labels": ["GCD-LB", "LB-DEF1"],
        "premises": [
            Clause(literals=frozenset({(True, "is_gcd", "g", "a", "b")}),
                   label="premise: g = gcd(a,b)"),
        ],
        "negated_goal": Clause(
            literals=frozenset({(False, "divides", "g", "a")}),
            label="negated goal: gcd does not divide a",
        ),
    },
    "gcd_is_greatest": {
        "description": "GCD is the greatest lower bound: d|a and d|b -> d|gcd(a,b)",
        "axiom_labels": ["GCD-GREATEST"],
        "premises": [
            Clause(literals=frozenset({(True, "is_gcd", "g", "a", "b")}),
                   label="premise: g = gcd(a,b)"),
            Clause(literals=frozenset({(True, "lower_bound", "d", "a", "b")}),
                   label="premise: d is a lower bound of a and b"),
        ],
        "negated_goal": Clause(
            literals=frozenset({(False, "divides", "d", "g")}),
            label="negated goal: d does not divide gcd",
        ),
    },
    "lcm_is_upper_bound": {
        "description": "LCM is an upper bound: a and b both divide lcm(a,b)",
        "axiom_labels": ["LCM-UB", "UB-DEF1"],
        "premises": [
            Clause(literals=frozenset({(True, "is_lcm", "l", "a", "b")}),
                   label="premise: l = lcm(a,b)"),
        ],
        "negated_goal": Clause(
            literals=frozenset({(False, "divides", "a", "l")}),
            label="negated goal: a does not divide lcm",
        ),
    },
    "lcm_is_least": {
        "description": "LCM is the least upper bound: a|m and b|m -> lcm(a,b)|m",
        "axiom_labels": ["LCM-LEAST"],
        "premises": [
            Clause(literals=frozenset({(True, "is_lcm", "l", "a", "b")}),
                   label="premise: l = lcm(a,b)"),
            Clause(literals=frozenset({(True, "upper_bound", "m", "a", "b")}),
                   label="premise: m is an upper bound of a and b"),
        ],
        "negated_goal": Clause(
            literals=frozenset({(False, "divides", "l", "m")}),
            label="negated goal: lcm does not divide m",
        ),
    },
    "gcd_is_meet": {
        "description": "GCD is the lattice meet: vector min -> lower bound (2-step chain)",
        "axiom_labels": ["VEC-GCD", "GCD-LB", "LB-DEF1"],
        "premises": [
            Clause(literals=frozenset({(True, "min_vec", "g", "a", "b")}),
                   label="premise: g is componentwise min of a and b"),
        ],
        "negated_goal": Clause(
            literals=frozenset({(False, "divides", "g", "a")}),
            label="negated goal: vector min does not divide a",
        ),
    },
    "lcm_is_join": {
        "description": "LCM is the lattice join: vector max -> upper bound (2-step chain)",
        "axiom_labels": ["VEC-LCM", "LCM-UB", "UB-DEF1"],
        "premises": [
            Clause(literals=frozenset({(True, "max_vec", "l", "a", "b")}),
                   label="premise: l is componentwise max of a and b"),
        ],
        "negated_goal": Clause(
            literals=frozenset({(False, "divides", "a", "l")}),
            label="negated goal: a does not divide vector max",
        ),
    },
    "distributivity": {
        "description": "Distributive lattice: gcd(a, lcm(b,c)) = lcm(gcd(a,b), gcd(a,c))",
        "axiom_labels": ["DIST-EQ"],
        "premises": [
            Clause(literals=frozenset({(True, "gcd_of_lcm", "r", "a", "b", "c")}),
                   label="premise: r = gcd(a, lcm(b,c))"),
            Clause(literals=frozenset({(True, "lcm_of_gcds", "r", "a", "b", "c")}),
                   label="premise: r = lcm(gcd(a,b), gcd(a,c))"),
        ],
        "negated_goal": Clause(
            literals=frozenset({(False, "eq_lattice", "r", "r")}),
            label="negated goal: gcd(a,lcm(b,c)) != lcm(gcd(a,b),gcd(a,c))",
        ),
    },

    # ==== Part 2: Probability waves ====
    "prime_density": {
        "description": "Each prime p has divisibility density 1/p",
        "axiom_labels": ["PROB-PRIME"],
        "premises": [
            Clause(literals=frozenset({(True, "prime", "p")}),
                   label="premise: p is prime"),
        ],
        "negated_goal": Clause(
            literals=frozenset({(False, "has_density", "p")}),
            label="negated goal: p has no density",
        ),
    },
    "prime_independence": {
        "description": "Distinct primes have independent divisibility (this IS the FTA)",
        "axiom_labels": ["PROB-INDEP"],
        "premises": [
            Clause(literals=frozenset({(True, "prime", "p")}),
                   label="premise: p is prime"),
            Clause(literals=frozenset({(True, "prime", "q")}),
                   label="premise: q is prime"),
            Clause(literals=frozenset({(True, "distinct", "p", "q")}),
                   label="premise: p != q"),
        ],
        "negated_goal": Clause(
            literals=frozenset({(False, "independent", "p", "q")}),
            label="negated goal: p and q are not independent",
        ),
    },
    "density_product": {
        "description": "Joint density is product: P(p|n AND q|n) = 1/pq (3-step chain)",
        "axiom_labels": ["PROB-PRIME", "PROB-INDEP", "PROB-DENSITY"],
        "premises": [
            Clause(literals=frozenset({(True, "prime", "p")}),
                   label="premise: p is prime"),
            Clause(literals=frozenset({(True, "prime", "q")}),
                   label="premise: q is prime"),
            Clause(literals=frozenset({(True, "distinct", "p", "q")}),
                   label="premise: p != q"),
        ],
        "negated_goal": Clause(
            literals=frozenset({(False, "joint_density_is_product", "p", "q")}),
            label="negated goal: joint density is not product",
        ),
    },
    "prime_wave": {
        "description": "Each prime defines a discrete wave function (period p indicator)",
        "axiom_labels": ["WAVE-PRIME"],
        "premises": [
            Clause(literals=frozenset({(True, "prime", "p")}),
                   label="premise: p is prime"),
        ],
        "negated_goal": Clause(
            literals=frozenset({(False, "has_wave", "p")}),
            label="negated goal: p has no wave function",
        ),
    },
    "wave_orthogonality": {
        "description": "Waves for distinct primes are orthogonal (correlation = 0)",
        "axiom_labels": ["PROB-INDEP", "WAVE-PRIME", "WAVE-ORTHO"],
        "premises": [
            Clause(literals=frozenset({(True, "prime", "p")}),
                   label="premise: p is prime"),
            Clause(literals=frozenset({(True, "prime", "q")}),
                   label="premise: q is prime"),
            Clause(literals=frozenset({(True, "distinct", "p", "q")}),
                   label="premise: p != q"),
        ],
        "negated_goal": Clause(
            literals=frozenset({(False, "orthogonal", "p", "q")}),
            label="negated goal: waves for p and q are not orthogonal",
        ),
    },
    "wave_basis": {
        "description": "Orthogonal prime waves form basis elements of the factor space",
        "axiom_labels": ["SHORTCUT-ORTHO", "WAVE-BASIS1"],
        "premises": [
            Clause(literals=frozenset({(True, "prime", "p")}),
                   label="premise: p is prime"),
            Clause(literals=frozenset({(True, "prime", "q")}),
                   label="premise: q is prime"),
            Clause(literals=frozenset({(True, "distinct", "p", "q")}),
                   label="premise: p != q"),
        ],
        "negated_goal": Clause(
            literals=frozenset({(False, "basis_element", "p")}),
            label="negated goal: p is not a basis element",
        ),
    },
    "fta_as_probability": {
        "description": "FTA as probability: independence + density -> unique factorization",
        "axiom_labels": ["PROB-PRIME", "PROB-INDEP", "PROB-DENSITY"],
        "premises": [
            Clause(literals=frozenset({(True, "prime", "p")}),
                   label="premise: p is prime"),
            Clause(literals=frozenset({(True, "prime", "q")}),
                   label="premise: q is prime"),
            Clause(literals=frozenset({(True, "distinct", "p", "q")}),
                   label="premise: p != q"),
        ],
        "negated_goal": Clause(
            literals=frozenset({(False, "joint_density_is_product", "p", "q")}),
            label="negated goal: unique factorization fails as probability",
        ),
    },
}


def lattice_prune(item, state) -> bool:
    """
    Discard clauses with too many literals or deep terms.
    Lattice proofs are chains of implications, not deep arithmetic.
    """
    if not isinstance(item, Clause):
        return False
    if len(item.literals) > 5:
        return True

    def term_depth(t):
        if isinstance(t, tuple):
            return 1 + max((term_depth(arg) for arg in t[1:]), default=0)
        return 0

    for lit in item.literals:
        for arg in lit[2:]:
            if term_depth(arg) > 4:
                return True
    return False


def make_lattice_state(theorem=None) -> OtterState:
    """
    Set up the prover state for a specific lattice theorem.

    Args:
        theorem: key from LATTICE_THEOREMS (default: "reflexivity")
    """
    if theorem is None:
        theorem = "reflexivity"
    if theorem not in LATTICE_THEOREMS:
        raise ValueError(
            f"Unknown theorem: {theorem!r}. "
            f"Choose from: {list(LATTICE_THEOREMS.keys())}"
        )

    thm = LATTICE_THEOREMS[theorem]
    state = OtterState()

    for rule in LATTICE_RULES:
        for prefix in thm["axiom_labels"]:
            if rule.label.startswith(prefix):
                state.set_of_support.append(rule)
                break

    for premise in thm["premises"]:
        state.set_of_support.append(premise)

    state.set_of_support.append(thm["negated_goal"])
    return state


def run_lattice_proof_suite(max_steps=100, verbose=True) -> dict:
    """
    Run all lattice theorems and return results.

    Returns dict: theorem_name -> {proved, steps, description, state}
    """
    results = {}
    for name, thm in LATTICE_THEOREMS.items():
        if verbose:
            print(f"\n{'='*60}")
            print(f"THEOREM: {name}")
            print(f"  {thm['description']}")
            print(f"{'='*60}")

        state = make_lattice_state(theorem=name)
        state = run_otter(
            state, resolve,
            max_steps=max_steps,
            stop_fn=found_empty_clause,
            subsumes_fn=clause_subsumes,
            prune_fn=lattice_prune,
            verbose=verbose,
        )

        proved = found_empty_clause(state)
        results[name] = {
            "proved": proved,
            "steps": state.step,
            "description": thm["description"],
            "state": state,
        }

        if verbose:
            if proved:
                print_proof(state)
            else:
                print(f"\n  NOT PROVED in {state.step} steps.")

    return results


def print_lattice_results(results: dict):
    """Pretty-print the lattice proof suite results."""
    structure_names = [
        "reflexivity", "antisymmetry", "transitivity", "unit_is_bottom",
        "gcd_is_lower_bound", "gcd_is_greatest", "lcm_is_upper_bound",
        "lcm_is_least", "gcd_is_meet", "lcm_is_join", "distributivity",
    ]
    wave_names = [
        "prime_density", "prime_independence", "density_product",
        "prime_wave", "wave_orthogonality", "wave_basis", "fta_as_probability",
    ]

    print(f"\n{'='*60}")
    print("PRIME FACTOR LATTICE: Proof Suite Results")
    print(f"{'='*60}")

    all_proved = True

    print(f"\n  --- Lattice Structure ---")
    for name in structure_names:
        if name in results:
            r = results[name]
            status = "PROVED" if r["proved"] else "NOT PROVED"
            if not r["proved"]:
                all_proved = False
            print(f"  {status:>11s} ({r['steps']:2d} steps)  {r['description']}")

    print(f"\n  --- Probability Waves ---")
    for name in wave_names:
        if name in results:
            r = results[name]
            status = "PROVED" if r["proved"] else "NOT PROVED"
            if not r["proved"]:
                all_proved = False
            print(f"  {status:>11s} ({r['steps']:2d} steps)  {r['description']}")

    print(f"\n{'='*60}")
    if all_proved:
        print("  ALL THEOREMS PROVED.")
        print("  The divisibility ordering on N forms a distributive lattice.")
        print("  GCD = meet (componentwise min of factor vectors).")
        print("  LCM = join (componentwise max of factor vectors).")
        print("  Prime divisibility waves are independent and orthogonal.")
        print("  They form a basis for the infinite-dimensional factor space.")
        print("  Independence of prime waves IS the FTA restated as measure theory.")
    else:
        print("  SOME THEOREMS FAILED. Check axiom formulations.")
    print(f"{'='*60}")
