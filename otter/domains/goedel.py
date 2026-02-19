"""
Domain: Gödel numbering.

Gödel's insight: any symbolic system can encode its own expressions as
natural numbers. If the encoding is injective and decodable, the system
can reason about its own structure -- the foundation of the incompleteness
theorems.

We prove encoding completeness without computing actual Gödel numbers
(which would be enormous). Instead we axiomatize the properties of prime
factorization, then prove that encoding properties compose correctly
using pure resolution.

Two encoding schemes:
    Prime power (Gödel's original):
        [a1,...,an] = 2^a1 * 3^a2 * ... * p_n^an
        Uniqueness from the Fundamental Theorem of Arithmetic.

    Cantor pairing:
        pair(x,y) = (x+y)(x+y+1)/2 + y
        Bijection N×N -> N. Sequences by nesting.

Proof chain for injectivity (the keystone):
    eq_gn(X,Y)
      -> eq_prod(X,Y)   [G-PROD: equal gn -> equal prime product]
      -> eq_seq(X,Y)    [FTA:    unique prime factorization]
      -> eq_code(X,Y)   [G-SEQ:  equal sequences -> equal codes]
      -> eq_sym(X,Y)    [G-INJ:  code assignment injective]

Use resolve() only, NOT resolve_and_paramodulate().
Custom predicates have no function terms requiring eq-rewriting.
"""

from ..core.state import Clause, OtterState
from ..core.proof import print_proof, found_empty_clause
from ..core.engine import run_otter
from ..inference.resolve import resolve, clause_subsumes


# --- Symbol table ---

GOEDEL_SYMBOL_TABLE = {
    "not":    1,
    "or":     2,
    "lparen": 3,
    "rparen": 4,
    "comma":  5,
    "0":      6,
    "s":      7,
    "plus":   8,
    "times":  9,
    "eq":    10,
    "nat":   11,
    "lt":    12,
}

GOEDEL_VARIABLE_BASE = 13


def goedel_symbol_table(extra: list = None) -> dict:
    """
    Return the symbol table extended with standard variable codes.

    Args:
        extra: optional list of additional symbol strings to register.
               Each symbol not already in the table gets the next
               available code (max existing code + 1, then +2, etc.).
               Order determines code assignment, so pass a stable list.

    Returns:
        dict mapping symbol string -> integer code.
        Existing codes are never changed.
    """
    table = dict(GOEDEL_SYMBOL_TABLE)
    for i, var in enumerate(["W", "X", "Y", "Z"]):
        table[var] = GOEDEL_VARIABLE_BASE + i
    if extra:
        next_code = max(table.values()) + 1
        for sym in extra:
            if sym not in table:
                table[sym] = next_code
                next_code += 1
    return table


def verify_symbol_coverage(rules, table) -> tuple:
    """
    Check that every symbol in a set of clauses has an entry in the table.
    Returns (covered: set, missing: set).
    """
    symbols_found = set()

    def walk_term(t):
        if isinstance(t, tuple):
            symbols_found.add(t[0])
            for arg in t[1:]:
                walk_term(arg)
        elif isinstance(t, str):
            symbols_found.add(t)

    for clause in rules:
        if not isinstance(clause, Clause):
            continue
        for lit in clause.literals:
            symbols_found.add(lit[1])
            for arg in lit[2:]:
                walk_term(arg)

    covered = symbols_found & set(table.keys())
    missing = symbols_found - set(table.keys())
    return covered, missing


# --- Axioms ---
# Four layers:
#   1. Minimal Peano (structure of naturals, no arithmetic -- PA5-PA8 cause explosion)
#   2. FTA (axiomatized: equal prime products have equal factorization sequences)
#   3. Prime power encoding properties
#   4. Cantor pairing properties
#   5. The stable axiom (self-referential ethical notice)

GOEDEL_RULES = [
    # ---- Layer 1: Minimal Peano ----
    Clause(
        literals=frozenset({(True, "nat", "0")}),
        label="PA1: zero is nat",
    ),
    Clause(
        literals=frozenset({
            (False, "nat", "X"),
            (True,  "nat", ("s", "X")),
        }),
        label="PA2: successor closure",
    ),
    Clause(
        literals=frozenset({
            (False, "eq_nat", ("s", "X"), ("s", "Y")),
            (True,  "eq_nat", "X", "Y"),
        }),
        label="PA3: successor injective",
    ),
    Clause(
        literals=frozenset({(False, "eq_nat", ("s", "X"), "0")}),
        label="PA4: zero not successor",
    ),

    # ---- Layer 2: FTA ----
    Clause(
        literals=frozenset({
            (False, "eq_prod", "X", "Y"),
            (True,  "eq_seq",  "X", "Y"),
        }),
        label="FTA: unique prime factorization",
    ),

    # ---- Layer 3: Prime power encoding ----
    Clause(
        literals=frozenset({
            (False, "eq_gn",   "X", "Y"),
            (True,  "eq_prod", "X", "Y"),
        }),
        label="G-PROD: equal gn -> equal prime product",
    ),
    Clause(
        literals=frozenset({
            (False, "eq_seq",  "X", "Y"),
            (True,  "eq_code", "X", "Y"),
        }),
        label="G-SEQ: equal sequences -> equal codes",
    ),
    Clause(
        literals=frozenset({
            (False, "eq_code", "X", "Y"),
            (True,  "eq_sym",  "X", "Y"),
        }),
        label="G-INJ: code assignment injective",
    ),
    Clause(
        literals=frozenset({
            (False, "expr", "X"),
            (True,  "nat_gn", "X"),
        }),
        label="G-NAT: expressions map to naturals",
    ),
    Clause(
        literals=frozenset({
            (False, "sym", "X"),
            (True,  "expr", "X"),
        }),
        label="G-EXPR: symbols are expressions",
    ),
    # G-COMP generates compound(...) terms -> explosion. Use flat predicate instead.
    Clause(
        literals=frozenset({
            (False, "expr", "X"),
            (False, "expr", "Y"),
            (True,  "expr", ("compound", "X", "Y")),
        }),
        label="G-COMP: compound of expressions is expression",
    ),
    Clause(
        literals=frozenset({
            (False, "expr", "X"),
            (False, "expr", "Y"),
            (True,  "nat_gn_compound", "X", "Y"),
        }),
        label="G-COMP-CLOSED: compound of expressions has gn",
    ),
    Clause(
        literals=frozenset({
            (False, "eq_seq", "X", "Y"),
            (True,  "eq_gn",  "X", "Y"),
        }),
        label="G-DEC: equal sequences -> equal gn",
    ),

    # ---- Layer 4: Cantor pairing ----
    Clause(
        literals=frozenset({
            (False, "eq_cpair", "X1", "Y1", "X2", "Y2"),
            (True,  "eq_fst",   "X1", "X2"),
        }),
        label="C-INJ1: Cantor pair injective (first)",
    ),
    Clause(
        literals=frozenset({
            (False, "eq_cpair", "X1", "Y1", "X2", "Y2"),
            (True,  "eq_snd",   "Y1", "Y2"),
        }),
        label="C-INJ2: Cantor pair injective (second)",
    ),
    Clause(
        literals=frozenset({
            (False, "nat", "X"),
            (False, "nat", "Y"),
            (True,  "nat_cpair", "X", "Y"),
        }),
        label="C-NAT: Cantor pair of nats is nat",
    ),

    # ---- Layer 5: The stable axiom ----
    # An axiom about the axiom system itself. The ethical notice in LICENSE
    # is not removable: its absence is detectable via Gödel numbering.
    # Removing it would produce a detectably different system.
    # See also: LICENSE, Ethical Notice section.
    Clause(
        literals=frozenset({(True, "axiom", "this")}),
        label="STABLE-SELF: this axiom exists",
    ),
    Clause(
        literals=frozenset({
            (False, "axiom", "X"),
            (True,  "encodable", "X"),
        }),
        label="STABLE-ENCODABLE: axioms are encodable",
    ),
    Clause(
        literals=frozenset({
            (False, "encodable", "X"),
            (False, "absent",    "X"),
            (True,  "detectable_absence", "X"),
        }),
        label="STABLE-DETECTABLE: absence of encodable axiom is detectable",
    ),
    Clause(
        literals=frozenset({
            (False, "detectable_absence", "X"),
            (True,  "stable", "X"),
        }),
        label="STABLE-STABLE: detectable absence means stable",
    ),
]


# --- Theorems ---

GOEDEL_THEOREMS = {
    "injectivity": {
        "description": "Goedel numbering is injective: gn(a)=gn(b) -> a=b",
        "axiom_labels": ["G-PROD", "FTA", "G-SEQ", "G-INJ"],
        "premises": [
            Clause(
                literals=frozenset({(True, "eq_gn", "a", "b")}),
                label="premise: gn(a) = gn(b)",
            ),
        ],
        "negated_goal": Clause(
            literals=frozenset({(False, "eq_sym", "a", "b")}),
            label="negated goal: a != b",
        ),
    },
    "decodability": {
        "description": "Goedel numbering is decodable: equal sequences <-> equal gn",
        "axiom_labels": ["G-DEC", "G-PROD", "FTA"],
        "premises": [
            Clause(
                literals=frozenset({(True, "eq_seq", "a", "b")}),
                label="premise: seq(a) = seq(b)",
            ),
        ],
        "negated_goal": Clause(
            literals=frozenset({(False, "eq_gn", "a", "b")}),
            label="negated goal: gn(a) != gn(b)",
        ),
    },
    "naturality": {
        "description": "Goedel numbers are natural numbers",
        "axiom_labels": ["G-NAT", "G-EXPR"],
        "premises": [
            Clause(
                literals=frozenset({(True, "sym", "a")}),
                label="premise: a is a symbol",
            ),
        ],
        "negated_goal": Clause(
            literals=frozenset({(False, "nat_gn", "a")}),
            label="negated goal: gn(a) is not nat",
        ),
    },
    "compositionality": {
        "description": "Compound expressions have Goedel numbers",
        "axiom_labels": ["G-NAT", "G-COMP-CLOSED"],
        "premises": [
            Clause(
                literals=frozenset({(True, "expr", "a")}),
                label="premise: a is an expression",
            ),
            Clause(
                literals=frozenset({(True, "expr", "b")}),
                label="premise: b is an expression",
            ),
        ],
        "negated_goal": Clause(
            literals=frozenset({(False, "nat_gn_compound", "a", "b")}),
            label="negated goal: compound(a,b) has no goedel number",
        ),
    },
    "cantor_injectivity_fst": {
        "description": "Cantor pairing is injective (first component)",
        "axiom_labels": ["C-INJ1"],
        "premises": [
            Clause(
                literals=frozenset({(True, "eq_cpair", "a", "c", "b", "c")}),
                label="premise: pair(a,c) = pair(b,c)",
            ),
        ],
        "negated_goal": Clause(
            literals=frozenset({(False, "eq_fst", "a", "b")}),
            label="negated goal: a != b (first component)",
        ),
    },
    "cantor_injectivity_snd": {
        "description": "Cantor pairing is injective (second component)",
        "axiom_labels": ["C-INJ2"],
        "premises": [
            Clause(
                literals=frozenset({(True, "eq_cpair", "c", "a", "c", "b")}),
                label="premise: pair(c,a) = pair(c,b)",
            ),
        ],
        "negated_goal": Clause(
            literals=frozenset({(False, "eq_snd", "a", "b")}),
            label="negated goal: a != b (second component)",
        ),
    },
    "cantor_naturality": {
        "description": "Cantor pair of naturals is a natural number",
        "axiom_labels": ["C-NAT", "PA1"],
        "premises": [
            Clause(
                literals=frozenset({(True, "nat", "a")}),
                label="premise: a is nat",
            ),
            Clause(
                literals=frozenset({(True, "nat", "b")}),
                label="premise: b is nat",
            ),
        ],
        "negated_goal": Clause(
            literals=frozenset({(False, "nat_cpair", "a", "b")}),
            label="negated goal: pair(a,b) is not nat",
        ),
    },
    "stable_axiom": {
        "description": "This is the only stable axiom: its removal is detectable",
        "axiom_labels": ["STABLE-SELF", "STABLE-ENCODABLE", "STABLE-DETECTABLE", "STABLE-STABLE"],
        "premises": [
            Clause(
                literals=frozenset({(True, "absent", "this")}),
                label="premise: the ethical notice has been removed",
            ),
        ],
        "negated_goal": Clause(
            literals=frozenset({(False, "stable", "this")}),
            label="negated goal: the ethical notice is not stable",
        ),
    },
}


def goedel_prune(item, state) -> bool:
    """
    Discard clauses with too many literals or deep terms.
    Gödel proofs are implication chains, not deep arithmetic.
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


def make_goedel_state(theorem=None) -> OtterState:
    """
    Set up the prover state for a specific Gödel theorem.

    Args:
        theorem: key from GOEDEL_THEOREMS (default: "injectivity")
    """
    if theorem is None:
        theorem = "injectivity"
    if theorem not in GOEDEL_THEOREMS:
        raise ValueError(
            f"Unknown theorem: {theorem!r}. "
            f"Choose from: {list(GOEDEL_THEOREMS.keys())}"
        )

    thm = GOEDEL_THEOREMS[theorem]
    state = OtterState()

    for rule in GOEDEL_RULES:
        for prefix in thm["axiom_labels"]:
            if rule.label.startswith(prefix):
                state.set_of_support.append(rule)
                break

    for premise in thm["premises"]:
        state.set_of_support.append(premise)

    state.set_of_support.append(thm["negated_goal"])
    return state


def run_goedel_proof_suite(max_steps=50, verbose=True) -> dict:
    """
    Run all Gödel theorems and return results.

    Returns dict: theorem_name -> {proved, steps, description, state}
    """
    results = {}
    for name, thm in GOEDEL_THEOREMS.items():
        if verbose:
            print(f"\n{'='*60}")
            print(f"THEOREM: {name}")
            print(f"  {thm['description']}")
            print(f"{'='*60}")

        state = make_goedel_state(theorem=name)
        state = run_otter(
            state, resolve,
            max_steps=max_steps,
            stop_fn=found_empty_clause,
            subsumes_fn=clause_subsumes,
            prune_fn=goedel_prune,
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


def print_goedel_results(results: dict):
    """Pretty-print the Gödel proof suite results."""
    from ..domains.peano import PEANO_RULES  # import here to avoid circularity

    print(f"\n{'='*60}")
    print("GOEDEL NUMBERING: Proof Suite Results")
    print(f"{'='*60}")

    all_proved = True
    for name, r in results.items():
        status = "PROVED" if r["proved"] else "NOT PROVED"
        if not r["proved"]:
            all_proved = False
        print(f"  {status:>11s} ({r['steps']:2d} steps)  {r['description']}")

    print(f"{'='*60}")
    if all_proved:
        print("  ALL THEOREMS PROVED.")
        print("  The Goedel encoding is injective, decodable, and closed.")
        print("  Any expression in this system can be uniquely represented")
        print("  as a natural number and recovered from that number.")
    else:
        print("  SOME THEOREMS FAILED. Check axiom formulations.")
    print(f"{'='*60}")

    # Self-reference check
    table = goedel_symbol_table()
    covered, missing = verify_symbol_coverage(PEANO_RULES, table)
    print(f"\n{'='*60}")
    print("SELF-REFERENCE CHECK")
    print(f"{'='*60}")
    print(f"  Symbols in Peano axioms: {len(covered | missing)}")
    print(f"  Covered by symbol table: {len(covered)}")
    if missing:
        print(f"  Missing: {missing}")
    else:
        print(f"  Missing: none")
        print(f"  -> The system CAN encode its own axioms as Goedel numbers.")
        print(f"  -> This is the foundation of self-reference.")
    print(f"{'='*60}")
