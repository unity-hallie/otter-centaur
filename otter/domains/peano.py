"""
Domain: Peano Arithmetic.

Proves arithmetic facts from first principles using resolution +
paramodulation. The default goal is 1+1=2.

Peano terms:
    "0"               -> zero
    ("s", "0")        -> 1 (successor of zero)
    ("s", ("s", "0")) -> 2
    ("plus", X, Y)    -> X + Y
    ("times", X, Y)   -> X * Y

Paramodulation is essential here. Pure resolution cannot handle
equational reasoning efficiently -- it requires explicit equality
axioms (reflexivity, symmetry, transitivity, congruence) which
generate combinatorial explosion. Paramodulation handles eq(s,t)
natively by rewriting subterms.

The pruning function is also essential: without it, the successor
function generates unbounded terms.
"""

from ..core.state import Clause, OtterState


PEANO_RULES = [
    # PA1: Zero is a natural number
    Clause(
        literals=frozenset({(True, "nat", "0")}),
        label="PA1: zero is nat",
    ),
    # PA2: Successor closure: nat(X) -> nat(s(X))
    Clause(
        literals=frozenset({
            (False, "nat", "X"),
            (True,  "nat", ("s", "X")),
        }),
        label="PA2: successor closure",
    ),
    # PA3: Successor injective: s(X) = s(Y) -> X = Y
    Clause(
        literals=frozenset({
            (False, "eq", ("s", "X"), ("s", "Y")),
            (True,  "eq", "X", "Y"),
        }),
        label="PA3: successor injective",
    ),
    # PA4: Zero is not a successor
    Clause(
        literals=frozenset({(False, "eq", ("s", "X"), "0")}),
        label="PA4: zero not successor",
    ),
    # PA5: Addition base: plus(X, 0) = X
    Clause(
        literals=frozenset({(True, "eq", ("plus", "X", "0"), "X")}),
        label="PA5: addition base",
    ),
    # PA6: Addition recursive: plus(X, s(Y)) = s(plus(X, Y))
    Clause(
        literals=frozenset({
            (True, "eq",
             ("plus", "X", ("s", "Y")),
             ("s", ("plus", "X", "Y"))),
        }),
        label="PA6: addition recursive",
    ),
    # PA7: Multiplication base: times(X, 0) = 0
    Clause(
        literals=frozenset({(True, "eq", ("times", "X", "0"), "0")}),
        label="PA7: multiplication base",
    ),
    # PA8: Multiplication recursive: times(X, s(Y)) = plus(times(X,Y), X)
    Clause(
        literals=frozenset({
            (True, "eq",
             ("times", "X", ("s", "Y")),
             ("plus", ("times", "X", "Y"), "X")),
        }),
        label="PA8: multiplication recursive",
    ),
    # Reflexivity as paramodulation seed
    Clause(
        literals=frozenset({(True, "eq", "X", "X")}),
        label="eq-refl: reflexivity",
    ),
]


def make_peano_state(goal_clauses=None) -> OtterState:
    """
    Set up Peano axioms with a goal.

    Default goal: 1 + 1 = 2
    i.e., ~eq(plus(s(0), s(0)), s(s(0)))
    """
    state = OtterState()

    for rule in PEANO_RULES:
        state.set_of_support.append(rule)

    if goal_clauses:
        for gc in goal_clauses:
            state.set_of_support.append(gc)
    else:
        state.set_of_support.append(Clause(
            literals=frozenset({
                (False, "eq",
                 ("plus", ("s", "0"), ("s", "0")),
                 ("s", ("s", "0"))),
            }),
            label="negated goal: 1+1 != 2",
        ))

    return state


def peano_prune(item, state) -> bool:
    """
    Discard clauses with deeply nested terms.

    Without this, Peano explodes: successor generates
    s(s(s(s(...)))) without bound. Depth 6 is enough to prove 1+1=2.
    """
    if not isinstance(item, Clause):
        return False

    def term_depth(t):
        if isinstance(t, tuple):
            return 1 + max((term_depth(arg) for arg in t[1:]), default=0)
        return 0

    for lit in item.literals:
        for arg in lit[2:]:
            if term_depth(arg) > 6:
                return True
    return False
