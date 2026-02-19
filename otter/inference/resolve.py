"""
Binary resolution: the core inference rule of the Otter loop.

Given two clauses, find a pair of complementary literals (one positive,
one negative, same predicate), unify their arguments, and produce a
resolvent that contains all the remaining literals from both clauses
with the unifying substitution applied.

If the resolvent is empty, a contradiction has been found.
"""

from ..core.state import Clause
from ..core.unification import (
    standardize_apart, unify_literals,
    apply_sub_to_clause,
)


def resolve(c1: Clause, c2: Clause) -> list:
    """
    Binary resolution between two clauses.

    Returns a list of resolvents (often 0 or 1, occasionally more
    if there are multiple complementary literal pairs).
    """
    lits1 = standardize_apart(c1.literals, "_L")
    lits2 = standardize_apart(c2.literals, "_R")

    results = []

    for lit1 in lits1:
        for lit2 in lits2:
            if lit1[0] == lit2[0]:
                continue  # same sign, can't resolve
            if lit1[1] != lit2[1]:
                continue  # different predicate

            sub = unify_literals(lit1, lit2)
            if sub is None:
                continue

            remaining1 = lits1 - {lit1}
            remaining2 = lits2 - {lit2}
            resolvent_lits = apply_sub_to_clause(sub, remaining1 | remaining2)

            results.append(Clause(
                literals=resolvent_lits,
                source=(c1.name, c2.name),
            ))

    return results


def clause_subsumes(c1: Clause, c2: Clause) -> bool:
    """
    c1 subsumes c2 if c1's literals are a proper subset of c2's.

    This is a simplified (literal-set) subsumption check. Full subsumption
    with unification is correct but expensive; this covers the common case
    where a shorter clause makes a longer one redundant.
    """
    if len(c1.literals) >= len(c2.literals):
        return False
    return c1.literals.issubset(c2.literals)
