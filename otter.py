"""
otter.py - Combinatorial search engine inspired by the Otter theorem prover.

Based on Johnicholas Hines' formulation of the Otter main loop as a
generalized algorithm that can run on adventure games, Little Alchemy,
Stern-Brocot trees, Turing machines, Zettelkasten, and more.

Extended by Hallie Larsson's insight that in an edge-first knowledge graph,
the primary objects are relationships (edges), not entities (nodes).
Nodes are just where edges meet. They don't have independent existence.

This implementation supports two modes:
1. Classic Otter: items are things, combination produces new things
2. Edge Otter: items are relationships (subject, predicate, object, confidence),
   combination means finding where two edges share a term and seeing
   what new relationship emerges from their intersection.

The combination function is pluggable. You can:
- Define it yourself (deterministic domain like Little Alchemy)
- Hand it to a human (interactive mode)
- Hand it to an LLM (API mode)
- Hand it to a human+LLM centaur (the interesting case)

Usage:
    python otter.py --domain little_alchemy    # classic demo
    python otter.py --domain edges             # edge-first demo
    python otter.py --domain interactive       # human in the loop
    python otter.py --domain llm              # LLM combination (needs API key)

See DOMAINS dict at bottom for how to define your own.

References:
- Otter theorem prover: https://www.cs.unm.edu/~mccune/otter/
- Johnicholas' email formulation (Jan 2026)
- Hallie's edge-first knowledge graphs / quantum-context
- Polysynthetic language structure (relationships as primary objects)
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Union
from collections import deque
from copy import deepcopy
import json
import sys
import os


# ============================================================
# Core data structures
# ============================================================

@dataclass
class Item:
    """An item in the Otter loop. Could be a thing or an edge."""
    name: str           # brief label (few words)
    content: str        # detailed content (paragraph)
    source: tuple = ()  # (parent_a_name, parent_b_name) if derived
    step: int = 0       # which step of the loop produced this

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, Item) and self.name == other.name

    def __repr__(self):
        return f"Item({self.name!r})"


@dataclass
class Edge:
    """A relationship in an edge-first graph. This IS the item."""
    subject: str
    predicate: str
    object: str
    confidence: float = 0.7  # confidence ceiling, not floor
    source: tuple = ()
    step: int = 0

    @property
    def name(self):
        return f"({self.subject} --{self.predicate}--> {self.object})"

    @property
    def content(self):
        return f"{self.subject} {self.predicate} {self.object} [confidence: {self.confidence}]"

    @property
    def terms(self):
        """All terms this edge touches."""
        return {self.subject, self.object}

    def shares_term_with(self, other: 'Edge') -> set:
        """Find shared terms between two edges."""
        return self.terms & other.terms

    def __hash__(self):
        return hash((self.subject, self.predicate, self.object))

    def __eq__(self, other):
        return (isinstance(other, Edge) and
                self.subject == other.subject and
                self.predicate == other.predicate and
                self.object == other.object)

    def __repr__(self):
        return f"Edge({self.name})"


# ============================================================
# The Otter loop
# ============================================================

@dataclass
class OtterState:
    """Full state of the Otter loop, serializable for continuity."""
    set_of_support: deque = field(default_factory=deque)  # never-focused items
    usable: list = field(default_factory=list)             # exhausted items
    history: list = field(default_factory=list)            # log of what happened
    step: int = 0
    halted: bool = False
    halt_reason: str = ""

    def to_dict(self):
        def serialize_literal(lit):
            """Serialize a literal tuple, handling nested tuples (functions)."""
            return [serialize_term(t) for t in lit]

        def serialize_term(t):
            if isinstance(t, tuple):
                return {"_fn": [serialize_term(x) for x in t]}
            if isinstance(t, bool):
                return {"_bool": t}
            return t

        def serialize(item):
            if isinstance(item, Clause):
                return {"type": "clause",
                        "literals": [serialize_literal(lit) for lit in item.literals],
                        "source": item.source, "step": item.step,
                        "label": item.label}
            elif isinstance(item, Edge):
                return {"type": "edge", "subject": item.subject,
                        "predicate": item.predicate, "object": item.object,
                        "confidence": item.confidence, "source": item.source,
                        "step": item.step}
            else:
                return {"type": "item", "name": item.name,
                        "content": item.content, "source": item.source,
                        "step": item.step}
        return {
            "set_of_support": [serialize(i) for i in self.set_of_support],
            "usable": [serialize(i) for i in self.usable],
            "history": self.history,
            "step": self.step,
            "halted": self.halted,
            "halt_reason": self.halt_reason,
        }

    @classmethod
    def from_dict(cls, d):
        def deserialize_term(t):
            if isinstance(t, dict):
                if "_fn" in t:
                    return tuple(deserialize_term(x) for x in t["_fn"])
                if "_bool" in t:
                    return t["_bool"]
            return t

        def deserialize_literal(parts):
            return tuple(deserialize_term(p) for p in parts)

        def deserialize(data):
            if data["type"] == "clause":
                lits = frozenset(deserialize_literal(lit) for lit in data["literals"])
                return Clause(lits, tuple(data.get("source", ())),
                             data.get("step", 0), data.get("label", ""))
            elif data["type"] == "edge":
                return Edge(data["subject"], data["predicate"],
                           data["object"], data.get("confidence", 0.7),
                           tuple(data.get("source", ())), data.get("step", 0))
            else:
                return Item(data["name"], data["content"],
                           tuple(data.get("source", ())), data.get("step", 0))
        state = cls()
        state.set_of_support = deque(deserialize(i) for i in d["set_of_support"])
        state.usable = [deserialize(i) for i in d["usable"]]
        state.history = d["history"]
        state.step = d["step"]
        state.halted = d.get("halted", False)
        state.halt_reason = d.get("halt_reason", "")
        return state

    def save(self, path="otter_state.json"):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path="otter_state.json"):
        with open(path) as f:
            return cls.from_dict(json.load(f))


def otter_step(
    state: OtterState,
    combine_fn: Callable,
    choose_focus_fn: Optional[Callable] = None,
    subsumes_fn: Optional[Callable] = None,
    prune_fn: Optional[Callable] = None,
    max_new_items: int = 50,
    verbose: bool = True,
) -> OtterState:
    """
    Execute one step of the Otter main loop.

    One step = pick a focus, try combining it with everything in usable,
    move focus to usable.

    Args:
        state: current OtterState
        combine_fn(x, y) -> list[Item|Edge]: try combining two items.
            Returns empty list if they don't combine.
            This is the pluggable heart of the whole thing.
        choose_focus_fn(set_of_support) -> Item|Edge: pick what to focus on.
            Default: FIFO (breadth-first). Override for human/LLM choice.
        subsumes_fn(a, b) -> bool: does a subsume b?
            Default: None (no subsumption checking).
        prune_fn(item, state) -> bool: should we skip this item?
            Default: None (no pruning).
        max_new_items: safety valve per step.
        verbose: print what's happening.

    Returns:
        Updated OtterState.
    """
    if not state.set_of_support:
        state.halted = True
        state.halt_reason = "set_of_support empty"
        return state

    # Choose focus
    if choose_focus_fn:
        focus = choose_focus_fn(state.set_of_support)
        state.set_of_support.remove(focus)
    else:
        focus = state.set_of_support.popleft()  # FIFO = breadth-first

    state.step += 1
    if verbose:
        print(f"\n--- Step {state.step}: Focus on {focus.name} ---")

    new_items = []

    # Try combining focus with everything in usable
    for y in state.usable:
        results = combine_fn(focus, y)
        for result in results:
            # Check if we already have this
            all_known = set(state.set_of_support) | set(state.usable) | set(new_items)
            if result in all_known:
                continue

            # Forward subsumption: skip if subsumed by something we have
            if subsumes_fn:
                if any(subsumes_fn(known, result) for known in all_known):
                    if verbose:
                        print(f"  [subsumed] {result.name}")
                    continue

            # Pruning
            if prune_fn and prune_fn(result, state):
                if verbose:
                    print(f"  [pruned] {result.name}")
                continue

            result.step = state.step
            new_items.append(result)
            if verbose:
                print(f"  [new] {result.name} (from {focus.name} + {y.name})")

            if len(new_items) >= max_new_items:
                if verbose:
                    print(f"  [safety valve] max_new_items reached")
                break
        if len(new_items) >= max_new_items:
            break

    # Back subsumption: do any new items subsume existing ones?
    if subsumes_fn:
        for new_item in new_items:
            to_remove = []
            for existing in list(state.set_of_support) + state.usable:
                if subsumes_fn(new_item, existing):
                    to_remove.append(existing)
            for item in to_remove:
                if item in state.set_of_support:
                    state.set_of_support.remove(item)
                if item in state.usable:
                    state.usable.remove(item)
                if verbose:
                    print(f"  [back-subsumed] {item.name} by {new_item.name}")

    # Move focus to usable
    state.usable.append(focus)

    # Add new items to set_of_support
    for item in new_items:
        state.set_of_support.append(item)

    # Log
    entry = {
        "step": state.step,
        "focus": focus.name,
        "combined_with": len(state.usable) - 1,  # minus the focus we just added
        "produced": [i.name for i in new_items],
        "set_of_support_size": len(state.set_of_support),
        "usable_size": len(state.usable),
    }
    state.history.append(entry)

    if verbose:
        print(f"  Set of support: {len(state.set_of_support)} | Usable: {len(state.usable)}")

    return state


def run_otter(
    state: OtterState,
    combine_fn: Callable,
    max_steps: int = 100,
    stop_fn: Optional[Callable] = None,
    save_path: Optional[str] = None,
    **kwargs,
) -> OtterState:
    """
    Run the Otter loop for multiple steps.

    Args:
        state: initial state
        combine_fn: combination function
        max_steps: safety limit
        stop_fn(state) -> bool: custom stopping condition
        save_path: if set, save state after each step
        **kwargs: passed to otter_step
    """
    for i in range(max_steps):
        if state.halted:
            break
        if stop_fn and stop_fn(state):
            state.halted = True
            state.halt_reason = "stop condition met"
            break
        state = otter_step(state, combine_fn, **kwargs)
        if save_path:
            state.save(save_path)
    return state


# ============================================================
# Domain: Little Alchemy (classic Otter with items)
# ============================================================

LITTLE_ALCHEMY_RULES = {
    frozenset({"air", "water"}): [("rain", "Water falling from sky")],
    frozenset({"air", "fire"}): [("energy", "Pure force")],
    frozenset({"air", "earth"}): [("dust", "Fine particles in the air")],
    frozenset({"earth", "water"}): [("mud", "Wet earth")],
    frozenset({"earth", "fire"}): [("lava", "Molten rock")],
    frozenset({"fire", "water"}): [("steam", "Heated water vapor")],
    frozenset({"rain", "earth"}): [("plant", "Growing life")],
    frozenset({"mud", "plant"}): [("swamp", "Wetland ecosystem")],
    frozenset({"lava", "water"}): [("stone", "Cooled volcanic rock"), ("steam", "Heated water vapor")],
    frozenset({"energy", "air"}): [("wind", "Moving air")],
    frozenset({"plant", "water"}): [("algae", "Simple aquatic life")],
    frozenset({"stone", "fire"}): [("metal", "Refined mineral")],
    frozenset({"stone", "air"}): [("sand", "Eroded stone")],
    frozenset({"sand", "fire"}): [("glass", "Transparent solid")],
    frozenset({"plant", "fire"}): [("tobacco", "Dried burning leaf"), ("ash", "Remnant of fire")],
    frozenset({"steam", "air"}): [("cloud", "Condensed vapor")],
    frozenset({"cloud", "water"}): [("rain", "Water falling from sky")],
    frozenset({"energy", "plant"}): [("tree", "Large plant")],
    frozenset({"tree", "fire"}): [("ash", "Remnant of fire"), ("charcoal", "Burned wood")],
}


def little_alchemy_combine(x: Item, y: Item) -> list:
    """Combination function for Little Alchemy domain."""
    key = frozenset({x.name, y.name})
    results = LITTLE_ALCHEMY_RULES.get(key, [])
    return [
        Item(name=name, content=desc, source=(x.name, y.name))
        for name, desc in results
    ]


def make_little_alchemy_state() -> OtterState:
    state = OtterState()
    for name, desc in [("air", "The atmosphere"), ("earth", "Solid ground"),
                       ("fire", "Combustion"), ("water", "H2O in liquid form")]:
        state.set_of_support.append(Item(name=name, content=desc))
    return state


# ============================================================
# Domain: Edge-first knowledge graph (Otter on relationships)
# ============================================================

def edge_combine(x: Edge, y: Edge) -> list:
    """
    Combine two edges by finding shared terms and producing new edges.

    When two relationships share a term, we can infer a new relationship
    between their non-shared terms, mediated by the shared term.

    Example:
        (alice --knows--> bob) + (bob --works_at--> acme)
        => (alice --connected_to--> acme) via bob

    Confidence of the new edge is the product of the parent confidences,
    capped at 0.7 (confidence ceiling - stay in sensing range).
    """
    shared = x.shares_term_with(y)
    if not shared:
        return []

    results = []
    for shared_term in shared:
        # Find the non-shared terms
        x_other = (x.terms - {shared_term}).pop() if len(x.terms - {shared_term}) > 0 else None
        y_other = (y.terms - {shared_term}).pop() if len(y.terms - {shared_term}) > 0 else None

        if x_other is None or y_other is None:
            continue
        if x_other == y_other:
            continue

        # New predicate is the composition
        new_predicate = f"{x.predicate}_via_{y.predicate}"

        # Confidence is product, capped
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
    Edge a subsumes edge b if they connect the same terms
    and a has higher or equal confidence.
    """
    return (a.subject == b.subject and
            a.object == b.object and
            a.confidence >= b.confidence and
            a.predicate != b.predicate)


SAMPLE_EDGES = [
    Edge("alice", "knows", "bob", 0.7),
    Edge("bob", "works_at", "acme", 0.7),
    Edge("acme", "builds", "widgets", 0.6),
    Edge("widgets", "require", "steel", 0.7),
    Edge("carol", "knows", "bob", 0.5),
    Edge("carol", "studies", "metallurgy", 0.7),
    Edge("metallurgy", "concerns", "steel", 0.7),
    Edge("alice", "studies", "design", 0.6),
    Edge("design", "shapes", "widgets", 0.5),
]


def make_edge_state() -> OtterState:
    state = OtterState()
    for edge in SAMPLE_EDGES:
        state.set_of_support.append(edge)
    return state


# ============================================================
# Domain: Resolution (symbolic reasoning -- the bone)
# ============================================================
#
# First-order resolution with unification. This is what makes Otter
# a theorem prover. The Clause is the symbolic skeleton; the Edge
# is the probabilistic flesh. A proven clause can stiffen an edge's
# confidence to 1.0. An edge network can suggest which clauses to
# even bother proving.
#
# Terms: variables (start with uppercase), constants/functions.
#   - "X", "Y"           -> variables
#   - "alice", "bob"      -> constants
#   - ("f", "X")          -> function application f(X)
#   - ("f", "a", "b")     -> f(a, b)
#
# Literals: (sign, predicate, arg1, arg2, ...)
#   - (True,  "knows", "alice", "bob")    ->  knows(alice, bob)
#   - (False, "knows", "alice", "bob")    -> ~knows(alice, bob)
#
# A Clause is a frozenset of literals (disjunction).
# The empty clause (frozenset()) is a contradiction -> proof found.

def is_variable(term):
    """Variables start with uppercase. Everything else is a constant or function."""
    if isinstance(term, str):
        return len(term) > 0 and term[0].isupper()
    return False


def is_function(term):
    """Functions are tuples: ("f", arg1, arg2, ...)."""
    return isinstance(term, tuple)


def occurs_in(var, term):
    """Does variable var occur anywhere in term? (Occurs check.)"""
    if var == term:
        return True
    if is_function(term):
        return any(occurs_in(var, arg) for arg in term[1:])
    return False


def apply_substitution(sub, term):
    """Apply a substitution dict to a term."""
    if is_variable(term):
        if term in sub:
            return apply_substitution(sub, sub[term])
        return term
    if is_function(term):
        return tuple([term[0]] + [apply_substitution(sub, arg) for arg in term[1:]])
    return term  # constant


def apply_sub_to_literal(sub, literal):
    """Apply substitution to a literal (sign, pred, arg1, arg2, ...)."""
    sign = literal[0]
    pred = literal[1]
    args = tuple(apply_substitution(sub, arg) for arg in literal[2:])
    return (sign, pred) + args


def apply_sub_to_clause(sub, clause):
    """Apply substitution to every literal in a clause."""
    return frozenset(apply_sub_to_literal(sub, lit) for lit in clause)


def unify_terms(t1, t2, sub=None):
    """
    Unify two terms under substitution sub.
    Returns updated substitution or None if unification fails.

    This is the Robinson unification algorithm with occurs check.
    """
    if sub is None:
        sub = {}

    t1 = apply_substitution(sub, t1)
    t2 = apply_substitution(sub, t2)

    if t1 == t2:
        return sub

    if is_variable(t1):
        if occurs_in(t1, t2):
            return None  # occurs check failure
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
            return None  # different function or arity
        for a1, a2 in zip(t1[1:], t2[1:]):
            sub = unify_terms(a1, a2, sub)
            if sub is None:
                return None
        return sub

    return None  # constant mismatch


def unify_literals(lit1, lit2, sub=None):
    """
    Unify two literals (ignoring sign). They must have the same predicate
    and arity. Returns substitution or None.
    """
    if lit1[1] != lit2[1]:
        return None  # different predicates
    if len(lit1) != len(lit2):
        return None  # different arity
    if sub is None:
        sub = {}
    for a1, a2 in zip(lit1[2:], lit2[2:]):
        sub = unify_terms(a1, a2, sub)
        if sub is None:
            return None
    return sub


def complement(literal):
    """Flip the sign of a literal."""
    return (not literal[0],) + literal[1:]


def standardize_apart(clause, suffix):
    """
    Rename all variables in a clause to avoid capture.
    Appends suffix to each variable name.
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


@dataclass
class Clause:
    """
    A disjunction of literals. The item type for resolution.

    The empty clause (literals = frozenset()) means contradiction / proof found.
    """
    literals: frozenset  # frozenset of (sign, pred, arg1, arg2, ...)
    source: tuple = ()
    step: int = 0
    label: str = ""      # human-readable label for the axiom/goal

    @property
    def name(self):
        if not self.literals:
            return "[]"  # empty clause = contradiction
        parts = []
        for lit in sorted(self.literals, key=str):
            sign = "" if lit[0] else "~"
            pred = lit[1]
            args = ", ".join(str(a) for a in lit[2:])
            parts.append(f"{sign}{pred}({args})")
        name = " | ".join(parts)
        if self.label:
            name = f"[{self.label}] {name}"
        return name

    @property
    def content(self):
        return self.name

    @property
    def is_empty(self):
        return len(self.literals) == 0

    def __hash__(self):
        return hash(self.literals)

    def __eq__(self, other):
        return isinstance(other, Clause) and self.literals == other.literals

    def __repr__(self):
        return f"Clause({self.name})"


def resolve(c1: Clause, c2: Clause) -> list:
    """
    Binary resolution: find complementary literals between c1 and c2,
    unify them, and produce resolvents.

    This is the combine_fn for the resolution domain.
    """
    # Standardize apart to avoid variable capture
    lits1 = standardize_apart(c1.literals, "_L")
    lits2 = standardize_apart(c2.literals, "_R")

    results = []

    for lit1 in lits1:
        for lit2 in lits2:
            # Check for complementary signs, same predicate
            if lit1[0] == lit2[0]:
                continue  # same sign, can't resolve
            if lit1[1] != lit2[1]:
                continue  # different predicate

            # Try to unify the arguments
            sub = unify_literals(lit1, lit2)
            if sub is None:
                continue

            # Build the resolvent: everything except the resolved pair
            remaining1 = lits1 - {lit1}
            remaining2 = lits2 - {lit2}
            resolvent_lits = apply_sub_to_clause(sub, remaining1 | remaining2)

            resolvent = Clause(
                literals=resolvent_lits,
                source=(c1.name, c2.name),
            )
            results.append(resolvent)

    return results


def clause_subsumes(c1: Clause, c2: Clause) -> bool:
    """
    Clause c1 subsumes c2 if c1 is a subset of c2 (after unification).
    Simplified: just check if c1's literals are a subset of c2's.
    Full subsumption with unification is expensive; this covers the common case.
    """
    if len(c1.literals) >= len(c2.literals):
        return False
    return c1.literals.issubset(c2.literals)


def found_empty_clause(state: OtterState) -> bool:
    """Stop condition: have we derived the empty clause?"""
    all_items = list(state.set_of_support) + state.usable
    return any(isinstance(c, Clause) and c.is_empty for c in all_items)


def extract_proof(state: OtterState) -> list:
    """
    Walk back from the empty clause through source links to extract
    the proof tree. Returns a list of (clause, depth) pairs.
    """
    # Find the empty clause
    empty = None
    all_items = {c.name: c for c in list(state.set_of_support) + state.usable
                 if isinstance(c, Clause)}
    for c in all_items.values():
        if c.is_empty:
            empty = c
            break
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
        src = ""
        if clause.source:
            src = f"  [from: {clause.source[0]} + {clause.source[1]}]"
        elif clause.label:
            src = f"  [axiom: {clause.label}]"
        print(f"  {i+1}. {indent}{clause.name}{src}")
    print(f"{'='*60}")
    print("  QED: empty clause derived -> negated goal is contradictory -> theorem holds.")


# --- Bridge: Clause <-> Edge ---

def clause_from_edge(edge: Edge) -> Clause:
    """
    Convert an edge to a clause.
    (alice --knows--> bob) becomes { knows(alice, bob) }

    The edge's confidence is not represented in the clause --
    that's the point. The clause is the bone. It says "this is
    structurally true" without hedging.
    """
    literal = (True, edge.predicate, edge.subject, edge.object)
    return Clause(
        literals=frozenset({literal}),
        label=f"from edge: {edge.name}",
    )


def edge_from_clause(clause: Clause, confidence=1.0) -> Optional[Edge]:
    """
    Convert a unit clause (single positive literal) back to an edge.
    A proven clause gets confidence 1.0 -- the bone is rigid.
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
    The bridge operation: take a set of uncertain edges and a set of
    proven clauses. Where a clause matches an edge's structure,
    stiffen its confidence to 1.0.

    This is where the bone meets the flesh.
    """
    proven_set = set()
    for clause in proven_clauses:
        for lit in clause.literals:
            if lit[0]:  # positive literal
                proven_set.add((lit[1],) + lit[2:])  # (pred, arg1, arg2, ...)

    result = []
    for edge in edges:
        key = (edge.predicate, edge.subject, edge.object)
        if key in proven_set:
            stiffened = Edge(
                edge.subject, edge.predicate, edge.object,
                confidence=1.0,
                source=edge.source,
                step=edge.step,
            )
            result.append(stiffened)
        else:
            result.append(edge)
    return result


# --- Sample problems ---

def make_resolution_state() -> OtterState:
    """
    Classic syllogism as a resolution problem.

    Axioms:
        1. All humans are mortal:    ~human(X) | mortal(X)
        2. Socrates is human:        human(socrates)

    Negated goal (for refutation):
        3. Socrates is NOT mortal:   ~mortal(socrates)

    Resolution should derive the empty clause, proving
    mortal(socrates).
    """
    state = OtterState()

    # Axiom: All humans are mortal.  forall X: human(X) -> mortal(X)
    # In clausal form: ~human(X) | mortal(X)
    state.set_of_support.append(Clause(
        literals=frozenset({
            (False, "human", "X"),
            (True, "mortal", "X"),
        }),
        label="all humans are mortal",
    ))

    # Fact: Socrates is human.
    state.set_of_support.append(Clause(
        literals=frozenset({(True, "human", "socrates")}),
        label="socrates is human",
    ))

    # Negated goal: Socrates is NOT mortal (we want to refute this).
    state.set_of_support.append(Clause(
        literals=frozenset({(False, "mortal", "socrates")}),
        label="negated goal: socrates not mortal",
    ))

    return state


def make_chain_resolution_state() -> OtterState:
    """
    A longer chain to show the engine working harder.

    Axioms:
        knows(alice, bob).
        trusts(X, Y) :- knows(X, Y).           [~knows(X,Y) | trusts(X,Y)]
        cooperates(X, Y) :- trusts(X, Y).       [~trusts(X,Y) | cooperates(X,Y)]
        builds_with(X, Y) :- cooperates(X, Y).  [~cooperates(X,Y) | builds_with(X,Y)]

    Negated goal: ~builds_with(alice, bob)

    Should prove: builds_with(alice, bob) via the chain.
    """
    state = OtterState()

    state.set_of_support.append(Clause(
        literals=frozenset({(True, "knows", "alice", "bob")}),
        label="alice knows bob",
    ))

    state.set_of_support.append(Clause(
        literals=frozenset({
            (False, "knows", "X", "Y"),
            (True, "trusts", "X", "Y"),
        }),
        label="knowing implies trusting",
    ))

    state.set_of_support.append(Clause(
        literals=frozenset({
            (False, "trusts", "X", "Y"),
            (True, "cooperates", "X", "Y"),
        }),
        label="trusting implies cooperating",
    ))

    state.set_of_support.append(Clause(
        literals=frozenset({
            (False, "cooperates", "X", "Y"),
            (True, "builds_with", "X", "Y"),
        }),
        label="cooperating implies building with",
    ))

    # Negated goal
    state.set_of_support.append(Clause(
        literals=frozenset({(False, "builds_with", "alice", "bob")}),
        label="negated goal: alice doesn't build with bob",
    ))

    return state


def make_bridge_demo_state() -> OtterState:
    """
    Demonstrate the bone-flesh bridge.

    Start with uncertain edges, convert some to clauses, prove something,
    then stiffen the edges that the proof supports.
    """
    state = OtterState()

    # The uncertain flesh (edges with confidence < 1.0)
    uncertain_edges = [
        Edge("alice", "knows", "bob", 0.7),
        Edge("bob", "works_at", "acme", 0.6),
        Edge("alice", "trusts", "bob", 0.3),  # low confidence -- but provable!
    ]

    # The symbolic bone (clauses that can prove trusts)
    # knows(X, Y) -> trusts(X, Y)
    state.set_of_support.append(Clause(
        literals=frozenset({
            (False, "knows", "X", "Y"),
            (True, "trusts", "X", "Y"),
        }),
        label="knowing implies trusting",
    ))

    # knows(alice, bob) -- asserted as certain (from edge, stiffened)
    state.set_of_support.append(Clause(
        literals=frozenset({(True, "knows", "alice", "bob")}),
        label="alice knows bob (from edge)",
    ))

    # negated goal: ~trusts(alice, bob)
    state.set_of_support.append(Clause(
        literals=frozenset({(False, "trusts", "alice", "bob")}),
        label="negated goal: alice doesn't trust bob",
    ))

    # Stash the edges in state metadata for the bridge demo
    state._uncertain_edges = uncertain_edges

    return state


# ============================================================
# Domain: Conditional certainty (uncertain axioms, rigid inference)
# ============================================================
#
# This is the real insight: axioms can be uncertain (edges with
# confidence 0.3, 0.7, whatever). But the INFERENCE is diamond-hard.
# "IF you accept these premises, THEN this conclusion follows with
# absolute logical necessity."
#
# The confidence of the conclusion = product of premise confidences.
# Not because the reasoning is uncertain -- the reasoning is perfect --
# but because the premises might not hold.
#
# A ConditionalProof wraps a symbolic proof with the confidence
# inherited from its uncertain axiom-edges.

@dataclass
class ConditionalProof:
    """A proof whose certainty is conditional on its axioms."""
    conclusion: str          # what was proven
    proof_steps: list        # the symbolic derivation (diamond-hard)
    axiom_confidences: dict  # {axiom_label: confidence} from the edges
    conditional_confidence: float  # product of axiom confidences

    @property
    def name(self):
        return f"[conf={self.conditional_confidence:.3f}] {self.conclusion}"

    def __repr__(self):
        return f"ConditionalProof({self.name})"


def prove_conditionally(edges: list, rules: list, goal_pred: str,
                        goal_subj: str, goal_obj: str,
                        max_steps: int = 50, verbose: bool = True):
    """
    Take uncertain edges and rigid inference rules.
    Build clauses from the edges, run resolution, and if a proof is found,
    compute the conditional confidence from the axiom-edges.

    Args:
        edges: list of Edge objects (uncertain axioms)
        rules: list of Clause objects (rigid inference rules, confidence=1.0)
        goal_pred, goal_subj, goal_obj: what to prove
        max_steps: resolution step limit

    Returns:
        ConditionalProof or None
    """
    state = OtterState()

    # Track which clauses came from which edges (and their confidence)
    axiom_map = {}  # clause_label -> edge.confidence

    # Convert edges to clauses (the uncertain premises)
    for edge in edges:
        clause = clause_from_edge(edge)
        axiom_map[clause.label] = edge.confidence
        state.set_of_support.append(clause)

    # Add the rigid inference rules
    for rule in rules:
        state.set_of_support.append(rule)

    # Add negated goal
    neg_goal = Clause(
        literals=frozenset({(False, goal_pred, goal_subj, goal_obj)}),
        label=f"negated goal: ~{goal_pred}({goal_subj}, {goal_obj})",
    )
    state.set_of_support.append(neg_goal)

    # Run resolution
    state = run_otter(
        state, resolve,
        max_steps=max_steps,
        stop_fn=found_empty_clause,
        subsumes_fn=clause_subsumes,
        verbose=verbose,
    )

    if not found_empty_clause(state):
        return None

    # Extract proof and compute conditional confidence
    proof = extract_proof(state)

    # Find which axioms were actually used in the proof
    used_axiom_labels = set()
    for clause, depth in proof:
        if clause.label and clause.label in axiom_map:
            used_axiom_labels.add(clause.label)
        # Also check parents
        for parent_name in clause.source:
            for c2, _ in proof:
                if c2.name == parent_name and c2.label in axiom_map:
                    used_axiom_labels.add(c2.label)

    # Walk the full derivation to find all leaf axioms
    all_items = {c.name: c for c in list(state.set_of_support) + state.usable
                 if isinstance(c, Clause)}

    def find_leaf_axioms(clause_name, visited=None):
        if visited is None:
            visited = set()
        if clause_name in visited:
            return set()
        visited.add(clause_name)
        if clause_name not in all_items:
            return set()
        clause = all_items[clause_name]
        if clause.label in axiom_map:
            return {clause.label}
        leaves = set()
        for parent in clause.source:
            leaves |= find_leaf_axioms(parent, visited)
        return leaves

    # Find the empty clause
    for c in all_items.values():
        if c.is_empty:
            used_axiom_labels = find_leaf_axioms(c.name)
            break

    used_confidences = {label: axiom_map[label] for label in used_axiom_labels
                        if label in axiom_map}

    # Conditional confidence = product of premise confidences
    if used_confidences:
        conditional = 1.0
        for conf in used_confidences.values():
            conditional *= conf
    else:
        conditional = 1.0  # no uncertain axioms used -> purely logical

    return ConditionalProof(
        conclusion=f"{goal_pred}({goal_subj}, {goal_obj})",
        proof_steps=proof,
        axiom_confidences=used_confidences,
        conditional_confidence=conditional,
    )


# ============================================================
# Domain: Peano Arithmetic (the foundation)
# ============================================================
#
# Peano axioms in clausal form. This is where it gets real.
# We can't encode the full induction schema in first-order logic
# (it's second-order), but we can encode specific induction instances
# and the basic structure of natural numbers.
#
# Terms:
#   "0"              -> zero (constant)
#   ("s", "0")       -> successor of zero = 1
#   ("s", ("s", "0")) -> 2
#   ("plus", X, Y)   -> addition function
#   ("times", X, Y)  -> multiplication function
#
# Predicates:
#   eq(X, Y)         -> X equals Y
#   nat(X)           -> X is a natural number
#   lt(X, Y)         -> X less than Y

def paramodulate(c1: Clause, c2: Clause) -> list:
    """
    Paramodulation: the equality-aware inference rule.

    Instead of axiomatizing equality (reflexivity, symmetry, transitivity,
    congruence -- which causes combinatorial explosion), paramodulation
    treats eq/2 specially:

    If c1 contains eq(s, t) and c2 contains a literal with subterm s',
    and s unifies with s', then replace s' with t in c2.

    This is what real theorem provers use for equational reasoning.
    It collapses what would take hundreds of resolution steps into one.
    """
    lits1 = standardize_apart(c1.literals, "_L")
    lits2 = standardize_apart(c2.literals, "_R")

    results = []

    # Try using equalities from c1 to rewrite c2
    for eq_lit in lits1:
        if not eq_lit[0] or eq_lit[1] != "eq" or len(eq_lit) != 4:
            continue  # need a positive eq(s, t)
        lhs, rhs = eq_lit[2], eq_lit[3]

        # Try rewriting subterms in c2's literals
        for target_lit in lits2:
            for arg_idx in range(2, len(target_lit)):
                for new_lit, sub in _paramod_at(target_lit, arg_idx,
                                                 lhs, rhs, {}):
                    if new_lit == target_lit:
                        continue
                    # Build resolvent: c1 minus eq_lit, c2 with rewritten lit
                    remaining1 = apply_sub_to_clause(sub, lits1 - {eq_lit})
                    remaining2 = apply_sub_to_clause(sub, lits2 - {target_lit})
                    new_lit_subbed = apply_sub_to_literal(sub, new_lit)
                    resolvent = Clause(
                        literals=remaining1 | remaining2 | frozenset({new_lit_subbed}),
                        source=(c1.name, c2.name),
                    )
                    results.append(resolvent)

    # Also try c2's equalities rewriting c1 (paramodulation is symmetric)
    for eq_lit in lits2:
        if not eq_lit[0] or eq_lit[1] != "eq" or len(eq_lit) != 4:
            continue
        lhs, rhs = eq_lit[2], eq_lit[3]

        for target_lit in lits1:
            for arg_idx in range(2, len(target_lit)):
                for new_lit, sub in _paramod_at(target_lit, arg_idx,
                                                 lhs, rhs, {}):
                    if new_lit == target_lit:
                        continue
                    remaining1 = apply_sub_to_clause(sub, lits1 - {target_lit})
                    remaining2 = apply_sub_to_clause(sub, lits2 - {eq_lit})
                    new_lit_subbed = apply_sub_to_literal(sub, new_lit)
                    resolvent = Clause(
                        literals=remaining1 | remaining2 | frozenset({new_lit_subbed}),
                        source=(c1.name, c2.name),
                    )
                    results.append(resolvent)

    return results


def _paramod_at(literal, arg_idx, lhs, rhs, sub):
    """
    Try to unify lhs with the subterm at arg_idx in literal,
    and yield the rewritten literal with rhs substituted.
    Also recurse into function subterms.
    """
    term = literal[arg_idx]
    # Try unifying the whole term
    new_sub = unify_terms(lhs, term, dict(sub))
    if new_sub is not None:
        new_term = apply_substitution(new_sub, rhs)
        new_literal = literal[:arg_idx] + (new_term,) + literal[arg_idx+1:]
        yield new_literal, new_sub

    # Recurse into function subterms
    if is_function(term):
        for i in range(1, len(term)):
            sub_term = term[i]
            new_sub = unify_terms(lhs, sub_term, dict(sub))
            if new_sub is not None:
                new_sub_term = apply_substitution(new_sub, rhs)
                new_func = term[:i] + (new_sub_term,) + term[i+1:]
                new_literal = literal[:arg_idx] + (new_func,) + literal[arg_idx+1:]
                yield new_literal, new_sub


def resolve_and_paramodulate(c1: Clause, c2: Clause) -> list:
    """
    Combined inference: resolution + paramodulation.
    This is what real equational theorem provers do.
    """
    return resolve(c1, c2) + paramodulate(c1, c2)


PEANO_RULES = [
    # 1. Zero is a natural number: nat(0)
    Clause(
        literals=frozenset({(True, "nat", "0")}),
        label="PA1: zero is nat",
    ),

    # 2. Successor closure: nat(X) -> nat(s(X))
    #    ~nat(X) | nat(s(X))
    Clause(
        literals=frozenset({
            (False, "nat", "X"),
            (True, "nat", ("s", "X")),
        }),
        label="PA2: successor closure",
    ),

    # 3. Successor injective: s(X) = s(Y) -> X = Y
    #    ~eq(s(X), s(Y)) | eq(X, Y)
    Clause(
        literals=frozenset({
            (False, "eq", ("s", "X"), ("s", "Y")),
            (True, "eq", "X", "Y"),
        }),
        label="PA3: successor injective",
    ),

    # 4. Zero is not a successor: ~eq(s(X), 0)
    Clause(
        literals=frozenset({(False, "eq", ("s", "X"), "0")}),
        label="PA4: zero not successor",
    ),

    # 5. Addition base: plus(X, 0) = X
    #    eq(plus(X, 0), X)
    Clause(
        literals=frozenset({(True, "eq", ("plus", "X", "0"), "X")}),
        label="PA5: addition base",
    ),

    # 6. Addition recursive: plus(X, s(Y)) = s(plus(X, Y))
    #    eq(plus(X, s(Y)), s(plus(X, Y)))
    Clause(
        literals=frozenset({
            (True, "eq", ("plus", "X", ("s", "Y")), ("s", ("plus", "X", "Y"))),
        }),
        label="PA6: addition recursive",
    ),

    # 7. Multiplication base: times(X, 0) = 0
    #    eq(times(X, 0), 0)
    Clause(
        literals=frozenset({(True, "eq", ("times", "X", "0"), "0")}),
        label="PA7: multiplication base",
    ),

    # 8. Multiplication recursive: times(X, s(Y)) = plus(times(X, Y), X)
    #    eq(times(X, s(Y)), plus(times(X, Y), X))
    Clause(
        literals=frozenset({
            (True, "eq", ("times", "X", ("s", "Y")),
             ("plus", ("times", "X", "Y"), "X")),
        }),
        label="PA8: multiplication recursive",
    ),

    # NOTE: Equality axioms (reflexivity, symmetry, transitivity) are NOT
    # needed here because paramodulation handles equality natively.
    # This avoids the combinatorial explosion that makes pure resolution
    # choke on equational reasoning.

    # Reflexivity is still useful as a "seed" for paramodulation:
    Clause(
        literals=frozenset({(True, "eq", "X", "X")}),
        label="eq-refl: reflexivity",
    ),
]


def make_peano_state(goal_clauses=None) -> OtterState:
    """
    Set up Peano arithmetic axioms with an optional goal.

    Default goal: prove 1 + 1 = 2
    i.e., eq(plus(s(0), s(0)), s(s(0)))
    Negated: ~eq(plus(s(0), s(0)), s(s(0)))
    """
    state = OtterState()

    for rule in PEANO_RULES:
        state.set_of_support.append(rule)

    if goal_clauses:
        for gc in goal_clauses:
            state.set_of_support.append(gc)
    else:
        # Default: prove 1 + 1 = 2
        # plus(s(0), s(0)) = s(s(0))
        # Negated goal: ~eq(plus(s(0), s(0)), s(s(0)))
        state.set_of_support.append(Clause(
            literals=frozenset({
                (False, "eq",
                 ("plus", ("s", "0"), ("s", "0")),
                 ("s", ("s", "0"))),
            }),
            label="negated goal: 1+1 != 2",
        ))

    return state


def peano_prune(item, state) -> bool:
    """
    Prune clauses that are getting too deep (term nesting).
    Without this, Peano arithmetic explodes -- the successor function
    generates unbounded terms. We cap nesting depth.
    """
    if not isinstance(item, Clause):
        return False

    def term_depth(t):
        if isinstance(t, tuple):
            return 1 + max((term_depth(arg) for arg in t[1:]), default=0)
        return 0

    for lit in item.literals:
        for arg in lit[2:]:
            if term_depth(arg) > 6:
                return True
    return False


# ============================================================
# Domain: Interactive (human in the loop)
# ============================================================

def interactive_choose_focus(set_of_support) -> 'Item|Edge':
    """Let the human choose what to focus on."""
    print("\nWhat would you like to focus on?")
    items = list(set_of_support)
    for i, item in enumerate(items):
        print(f"  [{i}] {item.name}")
    while True:
        try:
            choice = input("> ").strip()
            if choice.lower() in ('q', 'quit', 'exit'):
                raise KeyboardInterrupt
            idx = int(choice)
            return items[idx]
        except (ValueError, IndexError):
            print("Enter a number, or 'q' to quit.")


def interactive_combine(x, y) -> list:
    """Let the human decide what combining two items produces."""
    print(f"\nCombine: {x.name} + {y.name}")
    print(f"  {x.name}: {x.content}")
    print(f"  {y.name}: {y.content}")
    result = input("What does this produce? (empty to skip, name:description) > ").strip()
    if not result:
        return []

    parts = result.split(":", 1)
    name = parts[0].strip()
    desc = parts[1].strip() if len(parts) > 1 else name

    if isinstance(x, Edge):
        # Parse as edge: subject predicate object
        edge_parts = name.split()
        if len(edge_parts) >= 3:
            return [Edge(edge_parts[0], edge_parts[1], " ".join(edge_parts[2:]),
                        confidence=0.5, source=(x.name, y.name))]

    return [Item(name=name, content=desc, source=(x.name, y.name))]


# ============================================================
# Domain: LLM combination (stub for Claude API)
# ============================================================

def make_llm_combine(api_key=None, model="claude-sonnet-4-20250514"):
    """
    Returns a combination function that uses Claude to decide
    whether and how two items combine.

    Requires: pip install anthropic

    The prompt structure follows Johnicholas' formulation:
    - Show the LLM two items with their detailed content
    - Ask if they can be combined
    - If yes, get a description and brief name for the result
    """
    def llm_combine(x, y) -> list:
        try:
            import anthropic
        except ImportError:
            print("pip install anthropic --break-system-packages")
            return []

        client = anthropic.Anthropic(api_key=api_key)

        prompt = f"""You are participating in a combinatorial exploration.

You will be shown two items. Decide if they can be meaningfully combined
to produce something new -- a new idea, connection, or synthesis.

Item A: {x.name}
{x.content}

Item B: {y.name}
{y.content}

Can these be combined into something new and interesting?

If NO, respond with just: NO

If YES, respond with exactly two lines:
NAME: (brief name, a few words)
CONTENT: (one paragraph describing the combination)"""

        response = client.messages.create(
            model=model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )

        text = response.content[0].text.strip()

        if text.upper().startswith("NO"):
            return []

        lines = text.split("\n")
        name = ""
        content = ""
        for line in lines:
            if line.startswith("NAME:"):
                name = line[5:].strip()
            elif line.startswith("CONTENT:"):
                content = line[8:].strip()

        if name and content:
            if isinstance(x, Edge):
                # Try to parse as edge
                words = name.split()
                if len(words) >= 3:
                    return [Edge(words[0], words[1], " ".join(words[2:]),
                                confidence=0.5, source=(x.name, y.name))]
            return [Item(name=name, content=content, source=(x.name, y.name))]

        return []

    return llm_combine


# ============================================================
# Visualization / reporting
# ============================================================

def print_state(state: OtterState):
    """Print current state summary."""
    print(f"\n{'='*60}")
    print(f"Step: {state.step}")
    print(f"Set of support ({len(state.set_of_support)}):")
    for item in state.set_of_support:
        src = f" (from {item.source[0]} + {item.source[1]})" if item.source else ""
        print(f"  {item.name}{src}")
    print(f"Usable ({len(state.usable)}):")
    for item in state.usable:
        print(f"  {item.name}")
    print(f"{'='*60}")


def print_history(state: OtterState):
    """Print combination history as a tree."""
    print(f"\n{'='*60}")
    print("Combination history:")
    print(f"{'='*60}")
    for entry in state.history:
        produced = ", ".join(entry["produced"]) if entry["produced"] else "(nothing new)"
        print(f"  Step {entry['step']}: Focused on {entry['focus']} -> {produced}")


def export_dot(state: OtterState, path="otter_graph.dot"):
    """Export the derivation graph as a DOT file for visualization."""
    with open(path, "w") as f:
        f.write("digraph otter {\n")
        f.write("  rankdir=BT;\n")
        f.write("  node [shape=box, style=rounded];\n")

        all_items = list(state.usable) + list(state.set_of_support)
        for item in all_items:
            label = item.name.replace('"', '\\"')
            color = "lightblue" if item in state.set_of_support else "lightgray"
            f.write(f'  "{label}" [fillcolor={color}, style=filled];\n')
            if item.source:
                for parent in item.source:
                    parent_label = parent.replace('"', '\\"')
                    f.write(f'  "{parent_label}" -> "{label}";\n')
        f.write("}\n")
    print(f"Graph exported to {path}")


# ============================================================
# Main
# ============================================================

DOMAINS = {
    "little_alchemy": {
        "make_state": make_little_alchemy_state,
        "combine_fn": little_alchemy_combine,
        "description": "Classic Little Alchemy: combine elements to discover new ones",
    },
    "edges": {
        "make_state": make_edge_state,
        "combine_fn": edge_combine,
        "subsumes_fn": edge_subsumes,
        "description": "Edge-first knowledge graph: relationships combine via shared terms",
    },
    "resolution": {
        "make_state": make_resolution_state,
        "combine_fn": resolve,
        "subsumes_fn": clause_subsumes,
        "stop_fn": found_empty_clause,
        "description": "Symbolic resolution: prove theorems via refutation",
    },
    "chain": {
        "make_state": make_chain_resolution_state,
        "combine_fn": resolve,
        "subsumes_fn": clause_subsumes,
        "stop_fn": found_empty_clause,
        "description": "Multi-step resolution: prove a chain of implications",
    },
    "bridge": {
        "make_state": make_bridge_demo_state,
        "combine_fn": resolve,
        "subsumes_fn": clause_subsumes,
        "stop_fn": found_empty_clause,
        "description": "Bone-flesh bridge: prove symbolic facts, stiffen uncertain edges",
    },
    "peano": {
        "make_state": make_peano_state,
        "combine_fn": resolve_and_paramodulate,
        "subsumes_fn": clause_subsumes,
        "stop_fn": found_empty_clause,
        "prune_fn": peano_prune,
        "description": "Peano arithmetic: prove 1+1=2 from first principles",
    },
    "interactive": {
        "make_state": make_little_alchemy_state,  # default seed, override as needed
        "combine_fn": interactive_combine,
        "choose_focus_fn": interactive_choose_focus,
        "description": "Human in the loop: you choose focus and decide combinations",
    },
}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Otter combinatorial search")
    parser.add_argument("--domain", choices=list(DOMAINS.keys()) + ["llm"],
                       default="little_alchemy", help="Which domain to explore")
    parser.add_argument("--steps", type=int, default=20, help="Max steps")
    parser.add_argument("--save", type=str, default=None, help="Save state to file")
    parser.add_argument("--load", type=str, default=None, help="Load state from file")
    parser.add_argument("--dot", type=str, default=None, help="Export DOT graph")
    parser.add_argument("--quiet", action="store_true", help="Less output")
    args = parser.parse_args()

    if args.load:
        state = OtterState.load(args.load)
        print(f"Loaded state from {args.load} (step {state.step})")
    elif args.domain == "llm":
        state = make_little_alchemy_state()
    else:
        state = DOMAINS[args.domain]["make_state"]()

    if args.domain == "llm":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        combine_fn = make_llm_combine(api_key)
        kwargs = {}
    else:
        domain = DOMAINS[args.domain]
        combine_fn = domain["combine_fn"]
        kwargs = {}
        if "choose_focus_fn" in domain:
            kwargs["choose_focus_fn"] = domain["choose_focus_fn"]
        if "subsumes_fn" in domain:
            kwargs["subsumes_fn"] = domain["subsumes_fn"]
        if "prune_fn" in domain:
            kwargs["prune_fn"] = domain["prune_fn"]
    print(f"Domain: {args.domain}")
    print_state(state)

    stop_fn = domain.get("stop_fn") if args.domain != "llm" else None

    try:
        state = run_otter(
            state, combine_fn,
            max_steps=args.steps,
            stop_fn=stop_fn,
            save_path=args.save,
            verbose=not args.quiet,
            **kwargs,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.")

    print_state(state)
    print_history(state)

    # Resolution domains: show proof if found
    if args.domain in ("resolution", "chain", "bridge", "peano"):
        if found_empty_clause(state):
            print_proof(state)
        else:
            print("\nNo proof found (empty clause not derived).")

    # Bridge domain: demonstrate edge stiffening
    if args.domain == "bridge" and found_empty_clause(state):
        uncertain = getattr(state, '_uncertain_edges', None)
        if uncertain:
            all_clauses = [c for c in list(state.set_of_support) + state.usable
                          if isinstance(c, Clause)]
            stiffened = stiffen_edges(uncertain, all_clauses)
            print(f"\n{'='*60}")
            print("BRIDGE: Edge stiffening results")
            print(f"{'='*60}")
            for orig, stiff in zip(uncertain, stiffened):
                if orig.confidence != stiff.confidence:
                    print(f"  {orig.name}: {orig.confidence} -> {stiff.confidence}  [PROVEN]")
                else:
                    print(f"  {orig.name}: {orig.confidence}  [unchanged]")
            print(f"{'='*60}")

    if args.dot:
        export_dot(state, args.dot)

    if args.save:
        state.save(args.save)
        print(f"State saved to {args.save}")
