"""
Domain: Edge-first knowledge graph.

Relationships (edges) are the primary objects. When two edges share a term,
they can compose into a new relationship between their non-shared terms.

Example:
    (alice --knows--> bob) + (bob --works_at--> acme)
    => (alice --knows_via_works_at--> acme)  via the shared term "bob"

Confidence of the new edge is the product of the parent confidences,
capped at 0.7 (stay in sensing range -- don't overclaim).
"""

from ..core.state import Edge, OtterState


def edge_combine(x: Edge, y: Edge) -> list:
    shared = x.shares_term_with(y)
    if not shared:
        return []

    results = []
    for shared_term in shared:
        x_other = (x.terms - {shared_term}).pop() if (x.terms - {shared_term}) else None
        y_other = (y.terms - {shared_term}).pop() if (y.terms - {shared_term}) else None

        if x_other is None or y_other is None:
            continue
        if x_other == y_other:
            continue

        new_predicate = f"{x.predicate}_via_{y.predicate}"
        new_confidence = min(x.confidence * y.confidence, 0.7)

        results.append(Edge(
            subject=x_other,
            predicate=new_predicate,
            object=y_other,
            confidence=new_confidence,
            source=(x.name, y.name),
        ))

    return results


def edge_subsumes(a: Edge, b: Edge) -> bool:
    """
    Edge a subsumes edge b if they connect the same nodes and a is more confident.
    (Different predicate required -- otherwise they're the same edge.)
    """
    return (a.subject == b.subject and
            a.object == b.object and
            a.confidence >= b.confidence and
            a.predicate != b.predicate)


SAMPLE_EDGES = [
    Edge("alice",       "knows",       "bob",        0.7),
    Edge("bob",         "works_at",    "acme",        0.7),
    Edge("acme",        "builds",      "widgets",     0.6),
    Edge("widgets",     "require",     "steel",       0.7),
    Edge("carol",       "knows",       "bob",         0.5),
    Edge("carol",       "studies",     "metallurgy",  0.7),
    Edge("metallurgy",  "concerns",    "steel",       0.7),
    Edge("alice",       "studies",     "design",      0.6),
    Edge("design",      "shapes",      "widgets",     0.5),
]


def make_edge_state() -> OtterState:
    state = OtterState()
    for edge in SAMPLE_EDGES:
        state.set_of_support.append(edge)
    return state
