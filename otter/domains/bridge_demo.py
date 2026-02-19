"""
Domain: Bone-flesh bridge demo.

Demonstrates the connection between uncertain edges (flesh) and symbolic
proofs (bone). Start with low-confidence edges, run resolution to prove
a fact, then stiffen the matching edges to confidence 1.0.

The proof doesn't change the edges' truth -- they were always true or
not. What changes is our certainty. The proof turns a guess into a fact.
"""

from ..core.state import Edge, Clause, OtterState


def make_bridge_demo_state() -> OtterState:
    """
    Uncertain edges:
        alice --knows--> bob          (confidence 0.7)
        bob --works_at--> acme        (confidence 0.6)
        alice --trusts--> bob         (confidence 0.3)  <- low, but provable!

    Symbolic rules:
        knows(X, Y) -> trusts(X, Y)

    Negated goal: ~trusts(alice, bob)

    After proof: alice --trusts--> bob rises to confidence 1.0.
    """
    state = OtterState()

    uncertain_edges = [
        Edge("alice", "knows",    "bob",  0.7),
        Edge("bob",   "works_at", "acme", 0.6),
        Edge("alice", "trusts",   "bob",  0.3),  # low -- but provable
    ]

    state.set_of_support.append(Clause(
        literals=frozenset({
            (False, "knows", "X", "Y"),
            (True,  "trusts", "X", "Y"),
        }),
        label="knowing implies trusting",
    ))
    state.set_of_support.append(Clause(
        literals=frozenset({(True, "knows", "alice", "bob")}),
        label="alice knows bob (from edge)",
    ))
    state.set_of_support.append(Clause(
        literals=frozenset({(False, "trusts", "alice", "bob")}),
        label="negated goal: alice doesn't trust bob",
    ))

    # Stash uncertain edges as metadata for post-proof stiffening
    state._uncertain_edges = uncertain_edges

    return state
