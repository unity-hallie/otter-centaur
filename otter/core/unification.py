"""
Robinson unification algorithm with occurs check.

This is the logical foundation that everything else builds on.
Given two terms, find a substitution that makes them identical --
or report that no such substitution exists.

Terms:
    str starting with uppercase -> variable:  "X", "Foo"
    str starting with lowercase -> constant:  "0", "alice"
    tuple                       -> function:  ("s", "0"), ("plus", "X", "Y")

Substitutions are plain dicts: {"X": ("s", "0"), "Y": "alice"}
"""


def is_variable(term) -> bool:
    """Variables start with uppercase. Everything else is a constant or function."""
    return isinstance(term, str) and len(term) > 0 and term[0].isupper()


def is_function(term) -> bool:
    """Functions are tuples: (name, arg1, arg2, ...)."""
    return isinstance(term, tuple)


def occurs_in(var, term) -> bool:
    """Does variable var occur anywhere in term? Prevents infinite substitutions."""
    if var == term:
        return True
    if is_function(term):
        return any(occurs_in(var, arg) for arg in term[1:])
    return False


def apply_substitution(sub: dict, term):
    """Apply a substitution dict to a single term. Follows chains."""
    if is_variable(term):
        if term in sub:
            return apply_substitution(sub, sub[term])
        return term
    if is_function(term):
        return tuple([term[0]] + [apply_substitution(sub, arg) for arg in term[1:]])
    return term  # constant


def apply_sub_to_literal(sub: dict, literal: tuple) -> tuple:
    """Apply substitution to a literal (sign, pred, arg1, arg2, ...)."""
    sign = literal[0]
    pred = literal[1]
    args = tuple(apply_substitution(sub, arg) for arg in literal[2:])
    return (sign, pred) + args


def apply_sub_to_clause(sub: dict, clause) -> frozenset:
    """Apply substitution to every literal in a clause (frozenset of literals)."""
    return frozenset(apply_sub_to_literal(sub, lit) for lit in clause)


def unify_terms(t1, t2, sub=None):
    """
    Unify two terms under substitution sub.

    Returns the updated substitution dict, or None if unification fails.
    This is Robinson's algorithm (1965) with occurs check for soundness.
    """
    if sub is None:
        sub = {}

    t1 = apply_substitution(sub, t1)
    t2 = apply_substitution(sub, t2)

    if t1 == t2:
        return sub

    if is_variable(t1):
        if occurs_in(t1, t2):
            return None  # occurs check: X unify f(X) is unsound
        sub = dict(sub)
        sub[t1] = t2
        return sub

    if is_variable(t2):
        if occurs_in(t2, t1):
            return None
        sub = dict(sub)
        sub[t2] = t1
        return sub

    if is_function(t1) and is_function(t2):
        if t1[0] != t2[0] or len(t1) != len(t2):
            return None  # different functor or arity
        for a1, a2 in zip(t1[1:], t2[1:]):
            sub = unify_terms(a1, a2, sub)
            if sub is None:
                return None
        return sub

    return None  # two different constants


def unify_literals(lit1: tuple, lit2: tuple, sub=None):
    """
    Unify two literals, ignoring sign. Same predicate and arity required.
    Returns substitution or None.
    """
    if lit1[1] != lit2[1]:
        return None
    if len(lit1) != len(lit2):
        return None
    if sub is None:
        sub = {}
    for a1, a2 in zip(lit1[2:], lit2[2:]):
        sub = unify_terms(a1, a2, sub)
        if sub is None:
            return None
    return sub


def complement(literal: tuple) -> tuple:
    """Flip the sign of a literal."""
    return (not literal[0],) + literal[1:]


def standardize_apart(clause, suffix: str) -> frozenset:
    """
    Rename all variables in a clause by appending suffix.
    Prevents variable capture when two clauses share variable names.
    """
    var_map = {}

    def rename(term):
        if is_variable(term):
            if term not in var_map:
                var_map[term] = term + suffix
            return var_map[term]
        if is_function(term):
            return tuple([term[0]] + [rename(arg) for arg in term[1:]])
        return term

    return frozenset(
        (lit[0], lit[1]) + tuple(rename(arg) for arg in lit[2:])
        for lit in clause
    )
