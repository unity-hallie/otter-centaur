"""
Domain: Resolution (symbolic theorem proving).

Two sample problems demonstrating the Otter loop as a first-order
resolution prover.

resolution -- Socrates syllogism (mortal(socrates))
chain      -- multi-step implication chain (builds_with(alice, bob))
"""

from ..core.state import Clause, OtterState


def make_resolution_state() -> OtterState:
    """
    Classic syllogism as a resolution refutation problem.

    Axioms:
        all humans are mortal:  ~human(X) | mortal(X)
        socrates is human:       human(socrates)

    Negated goal (to refute):
        socrates is NOT mortal: ~mortal(socrates)

    Resolution derives the empty clause, proving mortal(socrates).
    """
    state = OtterState()

    state.set_of_support.append(Clause(
        literals=frozenset({
            (False, "human", "X"),
            (True,  "mortal", "X"),
        }),
        label="all humans are mortal",
    ))
    state.set_of_support.append(Clause(
        literals=frozenset({(True, "human", "socrates")}),
        label="socrates is human",
    ))
    state.set_of_support.append(Clause(
        literals=frozenset({(False, "mortal", "socrates")}),
        label="negated goal: socrates not mortal",
    ))

    return state


def make_chain_resolution_state() -> OtterState:
    """
    Multi-step implication chain.

    Axioms:
        knows(alice, bob).
        knows(X,Y)      -> trusts(X,Y)
        trusts(X,Y)     -> cooperates(X,Y)
        cooperates(X,Y) -> builds_with(X,Y)

    Negated goal: ~builds_with(alice, bob)

    Proves: builds_with(alice, bob) via the chain.
    """
    state = OtterState()

    state.set_of_support.append(Clause(
        literals=frozenset({(True, "knows", "alice", "bob")}),
        label="alice knows bob",
    ))
    state.set_of_support.append(Clause(
        literals=frozenset({
            (False, "knows", "X", "Y"),
            (True,  "trusts", "X", "Y"),
        }),
        label="knowing implies trusting",
    ))
    state.set_of_support.append(Clause(
        literals=frozenset({
            (False, "trusts", "X", "Y"),
            (True,  "cooperates", "X", "Y"),
        }),
        label="trusting implies cooperating",
    ))
    state.set_of_support.append(Clause(
        literals=frozenset({
            (False, "cooperates", "X", "Y"),
            (True,  "builds_with", "X", "Y"),
        }),
        label="cooperating implies building with",
    ))
    state.set_of_support.append(Clause(
        literals=frozenset({(False, "builds_with", "alice", "bob")}),
        label="negated goal: alice doesn't build with bob",
    ))

    return state
