"""
Paramodulation: equality-aware inference.

Instead of axiomatizing equality with reflexivity/symmetry/transitivity/
congruence axioms (which causes combinatorial explosion), paramodulation
treats eq/2 natively:

    If c1 contains a positive literal  eq(s, t),
    and c2 contains a subterm s' that unifies with s,
    then produce a new clause with s' replaced by t.

This is what real equational provers use. It collapses what would take
hundreds of resolution steps into one, and is necessary for any domain
that reasons about arithmetic or algebraic equality.

Used by: Peano arithmetic domain.
Not needed by: Gödel, Lattice (no function terms in their equality predicates).
"""

from ..core.state import Clause
from ..core.unification import (
    standardize_apart, unify_terms,
    apply_substitution, apply_sub_to_literal, apply_sub_to_clause,
    is_function,
)
from .resolve import resolve


def _paramod_at(literal, arg_idx, lhs, rhs, sub):
    """
    Try to unify lhs with the subterm at arg_idx in literal.
    Yield (rewritten_literal, substitution) pairs.
    Also recurses into function subterms.
    """
    term = literal[arg_idx]

    new_sub = unify_terms(lhs, term, dict(sub))
    if new_sub is not None:
        new_term = apply_substitution(new_sub, rhs)
        new_literal = literal[:arg_idx] + (new_term,) + literal[arg_idx+1:]
        yield new_literal, new_sub

    if is_function(term):
        for i in range(1, len(term)):
            sub_term = term[i]
            new_sub = unify_terms(lhs, sub_term, dict(sub))
            if new_sub is not None:
                new_sub_term = apply_substitution(new_sub, rhs)
                new_func = term[:i] + (new_sub_term,) + term[i+1:]
                new_literal = literal[:arg_idx] + (new_func,) + literal[arg_idx+1:]
                yield new_literal, new_sub


def paramodulate(c1: Clause, c2: Clause) -> list:
    """
    Paramodulation between two clauses.

    Tries equalities from c1 to rewrite c2, and equalities from c2
    to rewrite c1 (paramodulation is symmetric).
    """
    lits1 = standardize_apart(c1.literals, "_L")
    lits2 = standardize_apart(c2.literals, "_R")

    results = []

    # c1's equalities rewriting c2
    for eq_lit in lits1:
        if not eq_lit[0] or eq_lit[1] != "eq" or len(eq_lit) != 4:
            continue
        lhs, rhs = eq_lit[2], eq_lit[3]

        for target_lit in lits2:
            for arg_idx in range(2, len(target_lit)):
                for new_lit, sub in _paramod_at(target_lit, arg_idx, lhs, rhs, {}):
                    if new_lit == target_lit:
                        continue
                    remaining1 = apply_sub_to_clause(sub, lits1 - {eq_lit})
                    remaining2 = apply_sub_to_clause(sub, lits2 - {target_lit})
                    new_lit_subbed = apply_sub_to_literal(sub, new_lit)
                    results.append(Clause(
                        literals=remaining1 | remaining2 | frozenset({new_lit_subbed}),
                        source=(c1.name, c2.name),
                    ))

    # c2's equalities rewriting c1
    for eq_lit in lits2:
        if not eq_lit[0] or eq_lit[1] != "eq" or len(eq_lit) != 4:
            continue
        lhs, rhs = eq_lit[2], eq_lit[3]

        for target_lit in lits1:
            for arg_idx in range(2, len(target_lit)):
                for new_lit, sub in _paramod_at(target_lit, arg_idx, lhs, rhs, {}):
                    if new_lit == target_lit:
                        continue
                    remaining1 = apply_sub_to_clause(sub, lits1 - {target_lit})
                    remaining2 = apply_sub_to_clause(sub, lits2 - {eq_lit})
                    new_lit_subbed = apply_sub_to_literal(sub, new_lit)
                    results.append(Clause(
                        literals=remaining1 | remaining2 | frozenset({new_lit_subbed}),
                        source=(c1.name, c2.name),
                    ))

    return results


def resolve_and_paramodulate(c1: Clause, c2: Clause) -> list:
    """Combined inference: resolution + paramodulation. Used by Peano domain."""
    return resolve(c1, c2) + paramodulate(c1, c2)
