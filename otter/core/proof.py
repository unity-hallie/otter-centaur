"""
Proof extraction and display.

After the Otter loop finds the empty clause (contradiction), these
utilities walk back through the source links to recover the proof tree.
"""

from .state import OtterState, Clause


def found_empty_clause(state: OtterState) -> bool:
    """Stop condition: has the empty clause (contradiction) been derived?"""
    all_items = list(state.set_of_support) + state.usable
    return any(isinstance(c, Clause) and c.is_empty for c in all_items)


def extract_proof(state: OtterState) -> list:
    """
    Walk back from the empty clause through source links.
    Returns a list of (clause, depth) pairs, ordered from axioms to empty clause.
    """
    all_items = {c.name: c for c in list(state.set_of_support) + state.usable
                 if isinstance(c, Clause)}

    empty = next((c for c in all_items.values() if c.is_empty), None)
    if empty is None:
        return []

    proof = []
    visited = set()

    def walk(clause, depth):
        if clause.name in visited:
            return
        visited.add(clause.name)
        proof.append((clause, depth))
        for parent_name in clause.source:
            if parent_name in all_items:
                walk(all_items[parent_name], depth + 1)

    walk(empty, 0)
    proof.reverse()
    return proof


def print_proof(state: OtterState):
    """Pretty-print the proof tree."""
    proof = extract_proof(state)
    if not proof:
        print("No proof found.")
        return
    print(f"\n{'='*60}")
    print("PROOF (refutation)")
    print(f"{'='*60}")
    for i, (clause, depth) in enumerate(proof):
        indent = "  " * depth
        if clause.source:
            src = f"  [from: {clause.source[0]} + {clause.source[1]}]"
        elif clause.label:
            src = f"  [axiom: {clause.label}]"
        else:
            src = ""
        print(f"  {i+1}. {indent}{clause.name}{src}")
    print(f"{'='*60}")
    print("  QED: empty clause derived -> negated goal is contradictory -> theorem holds.")
