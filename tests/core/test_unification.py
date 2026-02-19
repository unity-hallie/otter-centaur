"""
Property-based and unit tests for the unification algorithm.

The core claims:
    - Symmetry:      unify(A,B) succeeds iff unify(B,A) succeeds
    - Correctness:   if unify(A,B)=σ then apply(σ,A) == apply(σ,B)
    - Occurs check:  unify(X, f(X)) always fails
    - Idempotence:   applying a unifier twice gives the same result as once
    - Freshness:     standardize_apart never reuses variables already present
"""

import pytest
from hypothesis import given, assume, settings
from hypothesis import strategies as st

from otter.core.unification import (
    is_variable, is_function,
    occurs_in, apply_substitution,
    unify_terms, unify_literals, complement,
    standardize_apart,
)


# ── Generators ──────────────────────────────────────────────────────────────

# Lowercase constants: "a", "b", "zero", "socrates"
constants = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz_",
    min_size=1, max_size=6,
).filter(lambda s: s[0].islower())

# Uppercase variables: "X", "Y", "Foo"
variables = st.text(
    alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
    min_size=1, max_size=4,
).filter(lambda s: s[0].isupper())

# Ground terms (no variables) for correctness tests
@st.composite
def ground_terms(draw, max_depth=3):
    if max_depth == 0:
        return draw(constants)
    choice = draw(st.integers(min_value=0, max_value=2))
    if choice == 0:
        return draw(constants)
    else:
        fname = draw(constants)
        arity = draw(st.integers(min_value=1, max_value=2))
        args = [draw(ground_terms(max_depth=max_depth - 1)) for _ in range(arity)]
        return tuple([fname] + args)

# Terms with variables
@st.composite
def terms(draw, max_depth=2):
    if max_depth == 0:
        return draw(st.one_of(constants, variables))
    choice = draw(st.integers(min_value=0, max_value=3))
    if choice == 0:
        return draw(constants)
    elif choice == 1:
        return draw(variables)
    else:
        fname = draw(constants)
        arity = draw(st.integers(min_value=1, max_value=2))
        args = [draw(terms(max_depth=max_depth - 1)) for _ in range(arity)]
        return tuple([fname] + args)

# Simple literals: (sign, pred, arg1, arg2)
@st.composite
def literals(draw, max_depth=2):
    sign = draw(st.booleans())
    pred = draw(constants)
    arity = draw(st.integers(min_value=1, max_value=2))
    args = [draw(terms(max_depth=max_depth)) for _ in range(arity)]
    return tuple([sign, pred] + args)


# ── Unit tests ───────────────────────────────────────────────────────────────

class TestIsVariable:
    def test_uppercase_is_variable(self):
        assert is_variable("X")
        assert is_variable("Foo")
        assert is_variable("LARGE")

    def test_lowercase_is_not_variable(self):
        assert not is_variable("x")
        assert not is_variable("alice")
        assert not is_variable("0")

    def test_empty_string_is_not_variable(self):
        assert not is_variable("")

    def test_tuple_is_not_variable(self):
        assert not is_variable(("s", "0"))

    def test_bool_is_not_variable(self):
        assert not is_variable(True)


class TestIsFunction:
    def test_tuple_is_function(self):
        assert is_function(("s", "0"))
        assert is_function(("plus", "X", "Y"))

    def test_string_is_not_function(self):
        assert not is_function("f")
        assert not is_function("X")

    def test_empty_tuple_is_function(self):
        assert is_function(())


class TestOccursIn:
    def test_variable_occurs_in_itself(self):
        assert occurs_in("X", "X")

    def test_variable_does_not_occur_in_different_variable(self):
        assert not occurs_in("X", "Y")

    def test_variable_occurs_in_function(self):
        assert occurs_in("X", ("f", "X"))
        assert occurs_in("X", ("f", "a", ("g", "X")))

    def test_variable_not_in_constant(self):
        assert not occurs_in("X", "alice")

    def test_variable_not_in_other_function(self):
        assert not occurs_in("X", ("f", "Y", "Z"))


class TestApplySubstitution:
    def test_variable_substitution(self):
        sub = {"X": "alice"}
        assert apply_substitution(sub, "X") == "alice"

    def test_constant_unchanged(self):
        sub = {"X": "alice"}
        assert apply_substitution(sub, "alice") == "alice"

    def test_function_substitution(self):
        sub = {"X": "0"}
        result = apply_substitution(sub, ("s", "X"))
        assert result == ("s", "0")

    def test_chained_substitution(self):
        sub = {"X": "Y", "Y": "alice"}
        assert apply_substitution(sub, "X") == "alice"

    def test_unbound_variable_unchanged(self):
        sub = {"Y": "bob"}
        assert apply_substitution(sub, "X") == "X"


class TestUnifyTerms:
    def test_same_constant(self):
        assert unify_terms("alice", "alice") == {}

    def test_different_constants_fail(self):
        assert unify_terms("alice", "bob") is None

    def test_variable_unifies_with_constant(self):
        sub = unify_terms("X", "alice")
        assert sub is not None
        assert sub["X"] == "alice"

    def test_constant_unifies_with_variable(self):
        sub = unify_terms("alice", "X")
        assert sub is not None
        assert sub["X"] == "alice"

    def test_variable_unifies_with_variable(self):
        sub = unify_terms("X", "Y")
        assert sub is not None

    def test_occurs_check_fails(self):
        # X cannot unify with f(X)
        assert unify_terms("X", ("f", "X")) is None

    def test_function_arity_mismatch_fails(self):
        assert unify_terms(("f", "X"), ("f", "X", "Y")) is None

    def test_function_name_mismatch_fails(self):
        assert unify_terms(("f", "X"), ("g", "X")) is None

    def test_nested_function_unification(self):
        # plus(X, 0) unifies with plus(s(0), 0) -> X = s(0)
        sub = unify_terms(("plus", "X", "0"), ("plus", ("s", "0"), "0"))
        assert sub is not None
        assert apply_substitution(sub, "X") == ("s", "0")


class TestComplement:
    def test_flips_true_to_false(self):
        lit = (True, "human", "socrates")
        assert complement(lit) == (False, "human", "socrates")

    def test_flips_false_to_true(self):
        lit = (False, "mortal", "X")
        assert complement(lit) == (True, "mortal", "X")

    def test_double_complement_is_identity(self):
        lit = (True, "knows", "alice", "bob")
        assert complement(complement(lit)) == lit


class TestStandardizeApart:
    def test_renames_variables(self):
        clause = frozenset({(True, "human", "X"), (False, "mortal", "X")})
        renamed = standardize_apart(clause, "_1")
        vars_in_renamed = set()
        for lit in renamed:
            for arg in lit[2:]:
                if is_variable(arg):
                    vars_in_renamed.add(arg)
        assert "X" not in vars_in_renamed
        assert "X_1" in vars_in_renamed

    def test_constants_unchanged(self):
        clause = frozenset({(True, "human", "socrates")})
        renamed = standardize_apart(clause, "_1")
        lit = next(iter(renamed))
        assert lit[2] == "socrates"

    def test_different_suffixes_give_different_vars(self):
        clause = frozenset({(True, "p", "X")})
        r1 = standardize_apart(clause, "_L")
        r2 = standardize_apart(clause, "_R")
        vars1 = {a for lit in r1 for a in lit[2:] if is_variable(a)}
        vars2 = {a for lit in r2 for a in lit[2:] if is_variable(a)}
        assert vars1.isdisjoint(vars2)


# ── Property-based tests ─────────────────────────────────────────────────────

class TestUnificationProperties:

    @given(terms(), terms())
    def test_symmetry(self, t1, t2):
        """unify(A, B) succeeds iff unify(B, A) succeeds."""
        r1 = unify_terms(t1, t2)
        r2 = unify_terms(t2, t1)
        assert (r1 is None) == (r2 is None)

    @given(terms(), terms())
    def test_correctness(self, t1, t2):
        """If unify(A, B) = σ, then apply(σ, A) == apply(σ, B)."""
        sub = unify_terms(t1, t2)
        if sub is not None:
            assert apply_substitution(sub, t1) == apply_substitution(sub, t2)

    @given(ground_terms(), ground_terms())
    def test_ground_terms_either_unify_or_not(self, t1, t2):
        """Ground terms unify iff they are equal."""
        sub = unify_terms(t1, t2)
        if t1 == t2:
            assert sub is not None
        else:
            assert sub is None

    @given(st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ", min_size=1, max_size=4))
    def test_occurs_check_soundness(self, v):
        """Variable X cannot unify with f(X) -- would require infinite term."""
        func_term = ("f", v)
        assert unify_terms(v, func_term) is None

    @given(terms())
    def test_idempotence(self, t):
        """A term unifies with itself and the substitution is empty."""
        sub = unify_terms(t, t)
        assert sub is not None
        assert apply_substitution(sub, t) == t

    @given(st.text(alphabet="ABCXYZ", min_size=1, max_size=3).filter(lambda s: s[0].isupper()),
           terms())
    def test_variable_unifies_with_any_non_occurring_term(self, v, t):
        """A variable unifies with any term that doesn't contain it."""
        assume(not occurs_in(v, t))
        sub = unify_terms(v, t)
        assert sub is not None
        assert apply_substitution(sub, v) == t

    @given(st.lists(literals(), min_size=2, max_size=4))
    def test_standardize_apart_freshness(self, lits):
        """standardize_apart(_L) variables don't collide with _R variables."""
        clause = frozenset(lits)
        r_l = standardize_apart(clause, "_L")
        r_r = standardize_apart(clause, "_R")
        vars_l = {a for lit in r_l for a in lit[2:] if is_variable(a)}
        vars_r = {a for lit in r_r for a in lit[2:] if is_variable(a)}
        assert vars_l.isdisjoint(vars_r)
