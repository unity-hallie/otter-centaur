"""
The bone-flesh bridge.

Clause is the symbolic bone: rigid, certain, no probability.
Edge is the probabilistic flesh: confident, uncertain, sensing.

These functions connect them:
  clause_from_edge   -- lift an uncertain edge into a certain clause
  edge_from_clause   -- recover an edge from a proven unit clause
  stiffen_edges      -- where proof meets flesh, confidence -> 1.0
  stiffen_to_limit   -- where convergent proof meets flesh, confidence -> L*
"""

from typing import Optional
from .state import Edge, Clause


def clause_from_edge(edge: Edge) -> Clause:
    """
    Lift an edge into a clause.

    (alice --knows--> bob) becomes { knows(alice, bob) }

    The confidence is left behind on purpose. The clause is the bone:
    it asserts structural truth without hedging.
    """
    literal = (True, edge.predicate, edge.subject, edge.object)
    return Clause(
        literals=frozenset({literal}),
        label=f"from edge: {edge.name}",
    )


def edge_from_clause(clause: Clause, confidence: float = 1.0) -> Optional[Edge]:
    """
    Recover an edge from a unit clause (single positive literal).

    A proven clause gets confidence 1.0 -- the bone is rigid.
    Returns None for multi-literal or negative clauses.
    """
    if len(clause.literals) != 1:
        return None
    lit = next(iter(clause.literals))
    if not lit[0]:  # negative literal
        return None
    if len(lit) < 4:  # need at least (sign, pred, subj, obj)
        return None
    return Edge(
        subject=str(lit[2]),
        predicate=lit[1],
        object=str(lit[3]),
        confidence=confidence,
        source=clause.source,
        step=clause.step,
    )


def stiffen_edges(edges: list, proven_clauses: list) -> list:
    """
    The bridge operation: where a clause proves what an edge asserts,
    raise that edge's confidence to 1.0.

    This is where the bone meets the flesh. The proof is certain;
    the edge that embodies that proof becomes certain too.
    """
    proven_set = set()
    for clause in proven_clauses:
        for lit in clause.literals:
            if lit[0]:  # positive literal
                proven_set.add((lit[1],) + lit[2:])

    result = []
    for edge in edges:
        key = (edge.predicate, edge.subject, edge.object)
        if key in proven_set:
            result.append(Edge(
                edge.subject, edge.predicate, edge.object,
                confidence=1.0,
                source=edge.source,
                step=edge.step,
            ))
        else:
            result.append(edge)
    return result


def stiffen_to_limit(edges: list, proven_clauses: list, limit: float) -> list:
    """
    The convergent bridge: where a convergent proof establishes a limit,
    raise matching edges' confidence to that limit.

    stiffen_edges raises to 1.0 — the classical case, where proof is
    absolute. stiffen_to_limit raises to L* — the convergent case,
    where L* is the proven maximum certainty the evidence can produce.

    If an edge's current confidence already exceeds the limit, it is
    left unchanged. The limit is a ceiling, not a floor.

    The claim: a confidence sequence whose limit is L*, proven to
    converge, IS L* at infinity. 0.999... = 1. The sequence reaching
    its limit is not approaching truth — it has arrived.
    """
    proven_set = set()
    for clause in proven_clauses:
        for lit in clause.literals:
            if lit[0]:  # positive literal
                proven_set.add((lit[1],) + lit[2:])

    result = []
    for edge in edges:
        key = (edge.predicate, edge.subject, edge.object)
        if key in proven_set and edge.confidence < limit:
            result.append(Edge(
                edge.subject, edge.predicate, edge.object,
                confidence=limit,
                source=edge.source,
                step=edge.step,
            ))
        else:
            result.append(edge)
    return result
