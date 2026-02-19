"""
Conditional certainty: uncertain axioms, rigid inference.

The insight: axioms can be uncertain (edges with confidence 0.3, 0.7, ...).
But the INFERENCE is diamond-hard. "IF you accept these premises, THEN
this conclusion follows with absolute logical necessity."

The confidence of the conclusion = product of the premise confidences.
Not because the reasoning is uncertain -- the reasoning is perfect --
but because the premises might not hold.

Guards against two attacks:
    1. Ex falso quodlibet: if the axioms are already contradictory,
       any conclusion can be "proven". We detect this and refuse.
    2. Confidence laundering: back-subsumption can delete axiom clauses
       from state, making it look like a proof used no uncertain premises.
       We track ALL clauses ever created to prevent this.
"""

from dataclasses import dataclass
from copy import deepcopy
from typing import Optional

from .core.state import Edge, Clause, OtterState
from .core.bridge import clause_from_edge
from .core.proof import found_empty_clause, extract_proof
from .core.engine import run_otter
from .inference.resolve import resolve, clause_subsumes


@dataclass
class ConditionalProof:
    """A proof whose certainty is conditional on its axiom-edges."""
    conclusion: str
    proof_steps: list
    axiom_confidences: dict   # {axiom_label: confidence}
    conditional_confidence: float  # product of axiom confidences

    @property
    def name(self):
        return f"[conf={self.conditional_confidence:.3f}] {self.conclusion}"

    def __repr__(self):
        return f"ConditionalProof({self.name})"


def prove_conditionally(
    edges: list,
    rules: list,
    goal_pred: str,
    goal_subj: str,
    goal_obj: str,
    max_steps: int = 50,
    verbose: bool = True,
) -> Optional[ConditionalProof]:
    """
    Prove a goal given uncertain edges and rigid rules.

    Args:
        edges:     list of Edge (uncertain axioms, confidence < 1.0)
        rules:     list of Clause (rigid inference rules)
        goal_pred, goal_subj, goal_obj: the predicate and arguments to prove
        max_steps: resolution step limit
        verbose:   print progress

    Returns:
        ConditionalProof if proved, None if not provable.
        Returns a zero-confidence ConditionalProof if axioms are inconsistent.
    """
    axiom_map = {}  # clause_label -> edge.confidence

    edge_clauses = []
    for edge in edges:
        clause = clause_from_edge(edge)
        axiom_map[clause.label] = edge.confidence
        edge_clauses.append(clause)

    # --- Guard 1: consistency check ---
    # Run resolution on edges + rules alone (no negated goal).
    # If they're already contradictory, any proof is worthless.
    consistency_state = OtterState()
    for c in edge_clauses:
        consistency_state.set_of_support.append(deepcopy(c))
    for r in rules:
        consistency_state.set_of_support.append(deepcopy(r))

    consistency_state = run_otter(
        consistency_state, resolve,
        max_steps=max_steps,
        stop_fn=found_empty_clause,
        subsumes_fn=clause_subsumes,
        verbose=False,
    )

    if found_empty_clause(consistency_state):
        if verbose:
            print("  [INCONSISTENT] Axioms are self-contradictory.")
            print("  Cannot trust any proof derived from inconsistent premises.")
            print("  (Ex falso quodlibet: from falsehood, anything follows.)")
        return ConditionalProof(
            conclusion=f"{goal_pred}({goal_subj}, {goal_obj})",
            proof_steps=[],
            axiom_confidences=axiom_map,
            conditional_confidence=0.0,
        )

    # --- Main proof attempt ---
    state = OtterState()

    # Index every clause we create. Back-subsumption can delete axioms from
    # state, making it appear the proof uses no uncertain premises (laundering).
    # Tracking all_clauses_ever prevents this.
    all_clauses_ever = {}

    for c in edge_clauses:
        state.set_of_support.append(c)
        all_clauses_ever[c.name] = c

    for r in rules:
        state.set_of_support.append(r)
        all_clauses_ever[r.name] = r

    neg_goal = Clause(
        literals=frozenset({(False, goal_pred, goal_subj, goal_obj)}),
        label=f"negated goal: ~{goal_pred}({goal_subj}, {goal_obj})",
    )
    state.set_of_support.append(neg_goal)
    all_clauses_ever[neg_goal.name] = neg_goal

    state = run_otter(
        state, resolve,
        max_steps=max_steps,
        stop_fn=found_empty_clause,
        subsumes_fn=clause_subsumes,
        verbose=verbose,
    )

    if not found_empty_clause(state):
        return None

    # Index all post-resolution clauses too
    for c in list(state.set_of_support) + state.usable:
        if isinstance(c, Clause):
            all_clauses_ever[c.name] = c

    # --- Walk the proof tree to find which axiom-edges were actually used ---
    def find_leaf_axioms(clause_name, visited=None):
        if visited is None:
            visited = set()
        if clause_name in visited:
            return set()
        visited.add(clause_name)
        if clause_name not in all_clauses_ever:
            return set()
        clause = all_clauses_ever[clause_name]
        if clause.label and clause.label in axiom_map:
            return {clause.label}
        if not clause.source:
            return set()
        leaves = set()
        for parent_name in clause.source:
            leaves |= find_leaf_axioms(parent_name, visited)
        return leaves

    used_axiom_labels = set()
    for c in all_clauses_ever.values():
        if c.is_empty:
            used_axiom_labels = find_leaf_axioms(c.name)
            break

    used_confidences = {
        label: axiom_map[label]
        for label in used_axiom_labels
        if label in axiom_map
    }

    if used_confidences:
        conditional = 1.0
        for conf in used_confidences.values():
            conditional *= conf
    else:
        conditional = 1.0  # no uncertain axioms used -> purely logical

    return ConditionalProof(
        conclusion=f"{goal_pred}({goal_subj}, {goal_obj})",
        proof_steps=extract_proof(state),
        axiom_confidences=used_confidences,
        conditional_confidence=conditional,
    )
