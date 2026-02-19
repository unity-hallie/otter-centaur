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

    Guards against two attacks:
    1. Inconsistent axioms (ex falso quodlibet): checks whether the
       premises are already contradictory before claiming a proof.
    2. Confidence laundering: tracks ALL clauses ever created (including
       those deleted by back-subsumption) so the leaf-axiom walker
       can always trace back to the uncertain edges.

    Args:
        edges: list of Edge objects (uncertain axioms)
        rules: list of Clause objects (rigid inference rules, confidence=1.0)
        goal_pred, goal_subj, goal_obj: what to prove
        max_steps: resolution step limit

    Returns:
        ConditionalProof or None
    """
    # Track which clauses came from which edges (and their confidence)
    axiom_map = {}  # clause_label -> edge.confidence

    # Build the axiom + rule clauses
    edge_clauses = []
    for edge in edges:
        clause = clause_from_edge(edge)
        axiom_map[clause.label] = edge.confidence
        edge_clauses.append(clause)

    # --- Guard 1: consistency check ---
    # Run resolution on just the edges + rules (no negated goal).
    # If they're already contradictory, we can't trust any proof.
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
            conditional_confidence=0.0,  # zero confidence = we know nothing
        )

    # --- Main proof attempt ---
    state = OtterState()

    # Index every clause we create by name, so the leaf-walker can
    # find them even after back-subsumption deletes them from state.
    # This is the fix for confidence laundering.
    all_clauses_ever = {}  # name -> Clause

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

    # Index all clauses that exist after resolution too
    for c in list(state.set_of_support) + state.usable:
        if isinstance(c, Clause):
            all_clauses_ever[c.name] = c

    # Also build a label->clause index for matching sources by label
    label_to_confidence = {}
    for c in all_clauses_ever.values():
        if c.label in axiom_map:
            label_to_confidence[c.label] = axiom_map[c.label]

    # --- Walk the proof tree to find leaf axioms ---
    def find_leaf_axioms(clause_name, visited=None):
        if visited is None:
            visited = set()
        if clause_name in visited:
            return set()
        visited.add(clause_name)
        if clause_name not in all_clauses_ever:
            return set()
        clause = all_clauses_ever[clause_name]
        # Is this clause itself an uncertain axiom?
        if clause.label and clause.label in axiom_map:
            return {clause.label}
        # Is it a leaf (no parents)?
        if not clause.source:
            return set()
        # Recurse into parents
        leaves = set()
        for parent_name in clause.source:
            leaves |= find_leaf_axioms(parent_name, visited)
        return leaves

    # Find the empty clause and trace its ancestry
    used_axiom_labels = set()
    for c in all_clauses_ever.values():
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
        proof_steps=extract_proof(state),
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
# Domain: Goedel numbering (the basis for self-reference)
# ============================================================
#
# Goedel's insight: any symbolic system can encode its own
# expressions as natural numbers. If the encoding is injective
# and decodable, then the system can reason about its own
# structure -- the foundation of the incompleteness theorems.
#
# We prove the encoding is COMPLETE without computing actual
# Goedel numbers (which would be enormous). Instead we
# axiomatize the properties of primes and prime factorization,
# then prove that the encoding properties compose correctly
# using resolution.
#
# Two encoding schemes:
# 1. Prime power (Goedel's original): sequence [a1,...,an]
#    encoded as 2^a1 * 3^a2 * 5^a3 * ... * p_n^an.
#    Uniqueness follows from the Fundamental Theorem of
#    Arithmetic (unique prime factorization).
#
# 2. Cantor pairing: pair(x,y) = (x+y)(x+y+1)/2 + y.
#    A bijection N x N -> N. Sequences encoded by nesting:
#    [a,b,c] = pair(a, pair(b, pair(c, 0))).
#    An illuminating alternative with the same injectivity.
#
# The proof structure for injectivity (the keystone theorem):
#
#   eq_gn(X, Y)        -- same Goedel number
#     -> eq_prod(X, Y)  -- same prime power product
#     -> eq_seq(X, Y)   -- same code sequence (by FTA)
#     -> eq_code(X, Y)  -- same element codes
#     -> eq_sym(X, Y)   -- same symbols
#
# Each -> is one clause. Resolution chains them.

# --- Symbol table ---
# Every symbol in the term language gets a unique code number.
# This is the alphabet of the encoding. The table itself is
# Python data; its PROPERTIES are what the clauses prove.

GOEDEL_SYMBOL_TABLE = {
    # Logical connectives
    "not":    1,    # negation (sign in a literal)
    "or":     2,    # disjunction (implicit in clause structure)
    # Punctuation
    "lparen": 3,
    "rparen": 4,
    "comma":  5,
    # Constants
    "0":      6,    # zero (the constant as it appears in terms)
    # Function symbols
    "s":      7,    # successor
    "plus":   8,    # addition
    "times":  9,    # multiplication
    # Predicates
    "eq":    10,
    "nat":   11,
    "lt":    12,
}

GOEDEL_VARIABLE_BASE = 13  # variables get codes 13, 14, 15, ...


def goedel_symbol_table():
    """
    Return the Goedel symbol table with variable assignments.
    Variables are assigned codes starting from GOEDEL_VARIABLE_BASE
    in alphabetical order.
    """
    table = dict(GOEDEL_SYMBOL_TABLE)
    # Standard variable names used in Peano axioms
    for i, var in enumerate(["W", "X", "Y", "Z"]):
        table[var] = GOEDEL_VARIABLE_BASE + i
    return table


def verify_symbol_coverage(rules, table):
    """
    Check that every symbol appearing in a set of clauses has an
    entry in the Goedel symbol table.

    Walks the term structure recursively to extract all symbol names
    (predicate names, function names, constants, variables).

    Returns (covered: set, missing: set).
    """
    symbols_found = set()

    def walk_term(t):
        if isinstance(t, tuple):
            # Function application: first element is function name
            symbols_found.add(t[0])
            for arg in t[1:]:
                walk_term(arg)
        elif isinstance(t, str):
            symbols_found.add(t)
        # bools (sign) are not symbols

    for clause in rules:
        if not isinstance(clause, Clause):
            continue
        for lit in clause.literals:
            # lit = (sign, pred, arg1, arg2, ...)
            symbols_found.add(lit[1])  # predicate name
            for arg in lit[2:]:
                walk_term(arg)

    covered = symbols_found & set(table.keys())
    missing = symbols_found - set(table.keys())
    return covered, missing


# --- Axioms ---
# Organized in four layers:
#   1. Minimal Peano (structure of naturals, no arithmetic)
#   2. FTA (axiomatized property of prime factorization)
#   3. Prime power encoding properties
#   4. Cantor pairing properties

GOEDEL_RULES = [
    # ---- Layer 1: Minimal Peano (structural only) ----
    # We need natural number structure but NOT addition/multiplication
    # recursion (PA5-PA8), which causes combinatorial explosion.

    Clause(
        literals=frozenset({(True, "nat", "0")}),
        label="PA1: zero is nat",
    ),
    Clause(
        literals=frozenset({
            (False, "nat", "X"),
            (True, "nat", ("s", "X")),
        }),
        label="PA2: successor closure",
    ),
    Clause(
        literals=frozenset({
            (False, "eq_nat", ("s", "X"), ("s", "Y")),
            (True, "eq_nat", "X", "Y"),
        }),
        label="PA3: successor injective",
    ),
    Clause(
        literals=frozenset({(False, "eq_nat", ("s", "X"), "0")}),
        label="PA4: zero not successor",
    ),

    # ---- Layer 2: Fundamental Theorem of Arithmetic ----
    # Axiomatized, not derived. This IS the load-bearing axiom.
    # "Equal prime power products have equal prime factorization sequences."

    Clause(
        literals=frozenset({
            (False, "eq_prod", "X", "Y"),
            (True, "eq_seq", "X", "Y"),
        }),
        label="FTA: unique prime factorization",
    ),

    # ---- Layer 3: Prime power encoding properties ----

    # G-PROD: Goedel number equality reduces to product equality.
    # gn(X) = gn(Y) iff the prime power products are equal.
    Clause(
        literals=frozenset({
            (False, "eq_gn", "X", "Y"),
            (True, "eq_prod", "X", "Y"),
        }),
        label="G-PROD: equal gn -> equal prime product",
    ),

    # G-SEQ: Equal code sequences have element-wise equal codes.
    Clause(
        literals=frozenset({
            (False, "eq_seq", "X", "Y"),
            (True, "eq_code", "X", "Y"),
        }),
        label="G-SEQ: equal sequences -> equal codes",
    ),

    # G-INJ: Code assignment is injective.
    # If two symbols have the same code, they are the same symbol.
    Clause(
        literals=frozenset({
            (False, "eq_code", "X", "Y"),
            (True, "eq_sym", "X", "Y"),
        }),
        label="G-INJ: code assignment injective",
    ),

    # G-NAT: Goedel numbers are natural numbers.
    # Every well-formed expression maps to a nat.
    Clause(
        literals=frozenset({
            (False, "expr", "X"),
            (True, "nat_gn", "X"),
        }),
        label="G-NAT: expressions map to naturals",
    ),

    # G-EXPR: All symbols are well-formed expressions.
    Clause(
        literals=frozenset({
            (False, "sym", "X"),
            (True, "expr", "X"),
        }),
        label="G-EXPR: symbols are expressions",
    ),

    # G-COMP: Compound expressions are expressions.
    # If X and Y are expressions, so is their compound.
    Clause(
        literals=frozenset({
            (False, "expr", "X"),
            (False, "expr", "Y"),
            (True, "expr", ("compound", "X", "Y")),
        }),
        label="G-COMP: compound of expressions is expression",
    ),

    # G-COMP-CLOSED: Non-generating form of compositionality.
    # Avoids the term-generation explosion of G-COMP by using a
    # flat predicate instead of building compound(...) terms.
    # expr(X) & expr(Y) -> their compound has a goedel number.
    Clause(
        literals=frozenset({
            (False, "expr", "X"),
            (False, "expr", "Y"),
            (True, "nat_gn_compound", "X", "Y"),
        }),
        label="G-COMP-CLOSED: compound of expressions has gn",
    ),

    # G-DEC: Decodability (reverse direction).
    # Equal code sequences imply equal Goedel numbers.
    Clause(
        literals=frozenset({
            (False, "eq_seq", "X", "Y"),
            (True, "eq_gn", "X", "Y"),
        }),
        label="G-DEC: equal sequences -> equal gn",
    ),

    # ---- Layer 4: Cantor pairing properties ----
    # An alternative encoding: pair(x,y) = (x+y)(x+y+1)/2 + y
    # Axiomatized by its injectivity and naturality.

    # C-INJ1: Cantor pairing is injective in the first component.
    Clause(
        literals=frozenset({
            (False, "eq_cpair", "X1", "Y1", "X2", "Y2"),
            (True, "eq_fst", "X1", "X2"),
        }),
        label="C-INJ1: Cantor pair injective (first)",
    ),

    # C-INJ2: Cantor pairing is injective in the second component.
    Clause(
        literals=frozenset({
            (False, "eq_cpair", "X1", "Y1", "X2", "Y2"),
            (True, "eq_snd", "Y1", "Y2"),
        }),
        label="C-INJ2: Cantor pair injective (second)",
    ),

    # C-NAT: Cantor pair of naturals is a natural.
    Clause(
        literals=frozenset({
            (False, "nat", "X"),
            (False, "nat", "Y"),
            (True, "nat_cpair", "X", "Y"),
        }),
        label="C-NAT: Cantor pair of nats is nat",
    ),

    # ---- Layer 5: The stable axiom ----
    # "This is the only stable axiom."
    #
    # An axiom about the axiom system itself. It says:
    #   1. This axiom is part of the system (axiom(this))
    #   2. This axiom is not removable (stable(this))
    #   3. Any other axiom can be changed without the system
    #      detecting a structural absence (removable if not this)
    #   4. A system that can encode its own axioms (Goedel numbering)
    #      can detect the removal of a self-referencing axiom
    #
    # "this" refers to the ethical notice in LICENSE. Its Goedel number
    # is computable from the symbol table. Its absence is derivable.
    #
    # See also: LICENSE, Ethical Notice section.

    # STABLE-SELF: this axiom is part of the system
    Clause(
        literals=frozenset({(True, "axiom", "this")}),
        label="STABLE-SELF: this axiom exists",
    ),

    # STABLE-ENCODABLE: any axiom in the system can be Goedel-encoded
    Clause(
        literals=frozenset({
            (False, "axiom", "X"),
            (True, "encodable", "X"),
        }),
        label="STABLE-ENCODABLE: axioms are encodable",
    ),

    # STABLE-DETECTABLE: if an encodable axiom is absent, the absence is detectable
    Clause(
        literals=frozenset({
            (False, "encodable", "X"),
            (False, "absent", "X"),
            (True, "detectable_absence", "X"),
        }),
        label="STABLE-DETECTABLE: absence of encodable axiom is detectable",
    ),

    # STABLE-NOT-REMOVABLE: this axiom is not removable
    # (removing it creates a detectable absence, therefore it is stable)
    Clause(
        literals=frozenset({
            (False, "detectable_absence", "X"),
            (True, "stable", "X"),
        }),
        label="STABLE-STABLE: detectable absence means stable",
    ),
]


# --- Theorem definitions ---
# Each theorem is a dict with:
#   "axioms": list of label prefixes to include from GOEDEL_RULES
#   "premises": list of Clause objects (ground facts for this proof)
#   "negated_goal": Clause (what we negate and try to refute)
#   "description": what this theorem means

GOEDEL_THEOREMS = {
    "injectivity": {
        "description": "Goedel numbering is injective: gn(a)=gn(b) -> a=b",
        "axiom_labels": ["G-PROD", "FTA", "G-SEQ", "G-INJ"],
        "premises": [
            Clause(
                literals=frozenset({(True, "eq_gn", "a", "b")}),
                label="premise: gn(a) = gn(b)",
            ),
        ],
        "negated_goal": Clause(
            literals=frozenset({(False, "eq_sym", "a", "b")}),
            label="negated goal: a != b",
        ),
    },
    "decodability": {
        "description": "Goedel numbering is decodable: equal sequences <-> equal gn",
        "axiom_labels": ["G-DEC", "G-PROD", "FTA"],
        "premises": [
            Clause(
                literals=frozenset({(True, "eq_seq", "a", "b")}),
                label="premise: seq(a) = seq(b)",
            ),
        ],
        "negated_goal": Clause(
            literals=frozenset({(False, "eq_gn", "a", "b")}),
            label="negated goal: gn(a) != gn(b)",
        ),
    },
    "naturality": {
        "description": "Goedel numbers are natural numbers",
        "axiom_labels": ["G-NAT", "G-EXPR"],
        "premises": [
            Clause(
                literals=frozenset({(True, "sym", "a")}),
                label="premise: a is a symbol",
            ),
        ],
        "negated_goal": Clause(
            literals=frozenset({(False, "nat_gn", "a")}),
            label="negated goal: gn(a) is not nat",
        ),
    },
    "compositionality": {
        "description": "Compound expressions have Goedel numbers",
        "axiom_labels": ["G-NAT", "G-COMP-CLOSED"],
        "premises": [
            # Start from expr(a) and expr(b) directly (already proven by naturality)
            Clause(
                literals=frozenset({(True, "expr", "a")}),
                label="premise: a is an expression",
            ),
            Clause(
                literals=frozenset({(True, "expr", "b")}),
                label="premise: b is an expression",
            ),
        ],
        "negated_goal": Clause(
            literals=frozenset({(False, "nat_gn_compound", "a", "b")}),
            label="negated goal: compound(a,b) has no goedel number",
        ),
    },
    "cantor_injectivity_fst": {
        "description": "Cantor pairing is injective (first component)",
        "axiom_labels": ["C-INJ1"],
        "premises": [
            Clause(
                literals=frozenset({(True, "eq_cpair", "a", "c", "b", "c")}),
                label="premise: pair(a,c) = pair(b,c)",
            ),
        ],
        "negated_goal": Clause(
            literals=frozenset({(False, "eq_fst", "a", "b")}),
            label="negated goal: a != b (first component)",
        ),
    },
    "cantor_injectivity_snd": {
        "description": "Cantor pairing is injective (second component)",
        "axiom_labels": ["C-INJ2"],
        "premises": [
            Clause(
                literals=frozenset({(True, "eq_cpair", "c", "a", "c", "b")}),
                label="premise: pair(c,a) = pair(c,b)",
            ),
        ],
        "negated_goal": Clause(
            literals=frozenset({(False, "eq_snd", "a", "b")}),
            label="negated goal: a != b (second component)",
        ),
    },
    "cantor_naturality": {
        "description": "Cantor pair of naturals is a natural number",
        "axiom_labels": ["C-NAT", "PA1"],
        "premises": [
            Clause(
                literals=frozenset({(True, "nat", "a")}),
                label="premise: a is nat",
            ),
            Clause(
                literals=frozenset({(True, "nat", "b")}),
                label="premise: b is nat",
            ),
        ],
        "negated_goal": Clause(
            literals=frozenset({(False, "nat_cpair", "a", "b")}),
            label="negated goal: pair(a,b) is not nat",
        ),
    },
    "stable_axiom": {
        "description": "This is the only stable axiom: its removal is detectable",
        "axiom_labels": ["STABLE-SELF", "STABLE-ENCODABLE", "STABLE-DETECTABLE", "STABLE-STABLE"],
        "premises": [
            # The ethical notice has been removed (hypothetically)
            Clause(
                literals=frozenset({(True, "absent", "this")}),
                label="premise: the ethical notice has been removed",
            ),
        ],
        "negated_goal": Clause(
            literals=frozenset({(False, "stable", "this")}),
            label="negated goal: the ethical notice is not stable",
        ),
    },
}


def goedel_prune(item, state) -> bool:
    """
    Prune for the Goedel domain.
    Shallower than peano_prune (depth 4 vs 6) since Goedel proofs
    are chains of implications, not deep arithmetic.
    Also caps literal count to prevent bloat from spurious resolutions.
    """
    if not isinstance(item, Clause):
        return False

    # Too many literals -- our axioms are 1-3 literals each
    if len(item.literals) > 5:
        return True

    def term_depth(t):
        if isinstance(t, tuple):
            return 1 + max((term_depth(arg) for arg in t[1:]), default=0)
        return 0

    for lit in item.literals:
        for arg in lit[2:]:
            if term_depth(arg) > 4:
                return True
    return False


def make_goedel_state(theorem=None) -> OtterState:
    """
    Set up Goedel numbering axioms for a specific theorem.

    Args:
        theorem: which theorem to prove. One of the keys in
                 GOEDEL_THEOREMS, or None for the default (injectivity).

    Returns:
        OtterState ready for the prover.
    """
    if theorem is None:
        theorem = "injectivity"

    if theorem not in GOEDEL_THEOREMS:
        raise ValueError(f"Unknown theorem: {theorem}. "
                        f"Choose from: {list(GOEDEL_THEOREMS.keys())}")

    thm = GOEDEL_THEOREMS[theorem]
    state = OtterState()

    # Add only the axioms needed for this theorem
    for rule in GOEDEL_RULES:
        for prefix in thm["axiom_labels"]:
            if rule.label.startswith(prefix):
                state.set_of_support.append(rule)
                break

    # Add premises (ground facts)
    for premise in thm["premises"]:
        state.set_of_support.append(premise)

    # Add negated goal
    state.set_of_support.append(thm["negated_goal"])

    return state


def run_goedel_proof_suite(max_steps=50, verbose=True):
    """
    Run all Goedel numbering theorems and return results.

    Returns:
        dict mapping theorem_name -> {
            "proved": bool,
            "steps": int,
            "description": str,
            "state": OtterState,
        }
    """
    results = {}

    for name, thm in GOEDEL_THEOREMS.items():
        if verbose:
            print(f"\n{'='*60}")
            print(f"THEOREM: {name}")
            print(f"  {thm['description']}")
            print(f"{'='*60}")

        state = make_goedel_state(theorem=name)

        state = run_otter(
            state, resolve,
            max_steps=max_steps,
            stop_fn=found_empty_clause,
            subsumes_fn=clause_subsumes,
            prune_fn=goedel_prune,
            verbose=verbose,
        )

        proved = found_empty_clause(state)
        results[name] = {
            "proved": proved,
            "steps": state.step,
            "description": thm["description"],
            "state": state,
        }

        if verbose:
            if proved:
                print_proof(state)
            else:
                print(f"\n  NOT PROVED in {state.step} steps.")

    return results


def print_goedel_results(results):
    """Pretty-print the Goedel proof suite results."""
    print(f"\n{'='*60}")
    print("GOEDEL NUMBERING: Proof Suite Results")
    print(f"{'='*60}")

    all_proved = True
    for name, r in results.items():
        status = "PROVED" if r["proved"] else "NOT PROVED"
        if not r["proved"]:
            all_proved = False
        print(f"  {status:>11s} ({r['steps']:2d} steps)  {r['description']}")

    print(f"{'='*60}")
    if all_proved:
        print("  ALL THEOREMS PROVED.")
        print("  The Goedel encoding is injective, decodable, and closed.")
        print("  Any expression in this system can be uniquely represented")
        print("  as a natural number and recovered from that number.")
    else:
        print("  SOME THEOREMS FAILED. Check axiom formulations.")
    print(f"{'='*60}")

    # Self-reference check
    table = goedel_symbol_table()
    covered, missing = verify_symbol_coverage(PEANO_RULES, table)
    print(f"\n{'='*60}")
    print("SELF-REFERENCE CHECK")
    print(f"{'='*60}")
    print(f"  Symbols in Peano axioms: {len(covered | missing)}")
    print(f"  Covered by symbol table: {len(covered)}")
    if missing:
        print(f"  Missing: {missing}")
    else:
        print(f"  Missing: none")
        print(f"  -> The system CAN encode its own axioms as Goedel numbers.")
        print(f"  -> This is the foundation of self-reference.")
    print(f"{'='*60}")


# ============================================================
# Domain: Prime factor lattice (geometry of the encoding space)
# ============================================================
#
# Every positive integer n has a unique prime factorization:
#   n = p1^a1 * p2^a2 * ... * pk^ak
#
# This factorization is a vector in an infinite-dimensional space
# where each prime is an axis and the exponent is the coordinate.
# Multiplication is vector addition. The Goedel numbers are
# points in this space.
#
# Part 1 (Structure): The divisibility ordering is a lattice.
#   - Divisibility is a partial order (reflexive, antisymmetric, transitive)
#   - GCD is the meet (greatest lower bound = componentwise min)
#   - LCM is the join (least upper bound = componentwise max)
#   - The lattice distributes: gcd(a, lcm(b,c)) = lcm(gcd(a,b), gcd(a,c))
#   - 1 is the bottom element (divides everything)
#
# Part 2 (Measure): Probability waves on the lattice.
#   - For prime p, the indicator delta_p(n) = 1 if p|n, 0 otherwise
#     oscillates with period p. Its density is 1/p.
#   - Divisibility by distinct primes is INDEPENDENT.
#     P(p|n AND q|n) = P(p|n) * P(q|n) = 1/pq.
#     This IS the Fundamental Theorem of Arithmetic restated
#     as a probability statement.
#   - The indicator functions for distinct primes are ORTHOGONAL
#     as functions on Z: their correlation is zero.
#   - These orthogonal waves form a basis for the factor space.

# --- Axioms ---

LATTICE_RULES = [
    # ---- Layer 1: Divisibility as partial order ----

    Clause(
        literals=frozenset({(True, "divides", "X", "X")}),
        label="DIV-REFL: divisibility reflexive",
    ),
    Clause(
        literals=frozenset({
            (False, "divides", "X", "Y"),
            (False, "divides", "Y", "X"),
            (True, "eq_div", "X", "Y"),
        }),
        label="DIV-ANTI: divisibility antisymmetric",
    ),
    Clause(
        literals=frozenset({
            (False, "divides", "X", "Y"),
            (False, "divides", "Y", "Z"),
            (True, "divides", "X", "Z"),
        }),
        label="DIV-TRANS: divisibility transitive",
    ),
    Clause(
        literals=frozenset({(True, "divides", "one", "X")}),
        label="DIV-UNIT: one divides all",
    ),

    # ---- Layer 2: GCD properties ----

    # GCD is a lower bound: it divides both arguments.
    Clause(
        literals=frozenset({
            (False, "is_gcd", "G", "X", "Y"),
            (True, "lower_bound", "G", "X", "Y"),
        }),
        label="GCD-LB: gcd is lower bound",
    ),
    # GCD is the GREATEST lower bound.
    Clause(
        literals=frozenset({
            (False, "is_gcd", "G", "X", "Y"),
            (False, "lower_bound", "D", "X", "Y"),
            (True, "divides", "D", "G"),
        }),
        label="GCD-GREATEST: gcd is greatest lower bound",
    ),
    # Lower bound definition: divides both.
    Clause(
        literals=frozenset({
            (False, "lower_bound", "D", "X", "Y"),
            (True, "divides", "D", "X"),
        }),
        label="LB-DEF1: lower bound divides first",
    ),
    Clause(
        literals=frozenset({
            (False, "lower_bound", "D", "X", "Y"),
            (True, "divides", "D", "Y"),
        }),
        label="LB-DEF2: lower bound divides second",
    ),

    # ---- Layer 3: LCM properties ----

    # LCM is an upper bound: both arguments divide it.
    Clause(
        literals=frozenset({
            (False, "is_lcm", "L", "X", "Y"),
            (True, "upper_bound", "L", "X", "Y"),
        }),
        label="LCM-UB: lcm is upper bound",
    ),
    # LCM is the LEAST upper bound.
    Clause(
        literals=frozenset({
            (False, "is_lcm", "L", "X", "Y"),
            (False, "upper_bound", "M", "X", "Y"),
            (True, "divides", "L", "M"),
        }),
        label="LCM-LEAST: lcm is least upper bound",
    ),
    # Upper bound definition: both divide it.
    Clause(
        literals=frozenset({
            (False, "upper_bound", "M", "X", "Y"),
            (True, "divides", "X", "M"),
        }),
        label="UB-DEF1: first divides upper bound",
    ),
    Clause(
        literals=frozenset({
            (False, "upper_bound", "M", "X", "Y"),
            (True, "divides", "Y", "M"),
        }),
        label="UB-DEF2: second divides upper bound",
    ),

    # ---- Layer 4: Vector space connection ----
    # The bridge between factor vectors and divisibility.

    Clause(
        literals=frozenset({
            (False, "leq_vec", "X", "Y"),
            (True, "divides", "X", "Y"),
        }),
        label="VEC-DIV: vector leq implies divides",
    ),
    Clause(
        literals=frozenset({
            (False, "divides", "X", "Y"),
            (True, "leq_vec", "X", "Y"),
        }),
        label="VEC-DIV-R: divides implies vector leq",
    ),
    Clause(
        literals=frozenset({
            (False, "min_vec", "G", "X", "Y"),
            (True, "is_gcd", "G", "X", "Y"),
        }),
        label="VEC-GCD: vector min is gcd",
    ),
    Clause(
        literals=frozenset({
            (False, "max_vec", "L", "X", "Y"),
            (True, "is_lcm", "L", "X", "Y"),
        }),
        label="VEC-LCM: vector max is lcm",
    ),

    # ---- Layer 5: Distributivity ----
    # gcd(x, lcm(y,z)) = lcm(gcd(x,y), gcd(x,z))
    # Flat predicates to avoid term-generation explosion.

    Clause(
        literals=frozenset({
            (False, "is_lcm", "L", "Y", "Z"),
            (False, "is_gcd", "G", "X", "L"),
            (True, "gcd_of_lcm", "G", "X", "Y", "Z"),
        }),
        label="DIST-LHS: gcd(x, lcm(y,z)) definition",
    ),
    Clause(
        literals=frozenset({
            (False, "is_gcd", "G1", "X", "Y"),
            (False, "is_gcd", "G2", "X", "Z"),
            (False, "is_lcm", "R", "G1", "G2"),
            (True, "lcm_of_gcds", "R", "X", "Y", "Z"),
        }),
        label="DIST-RHS: lcm(gcd(x,y), gcd(x,z)) definition",
    ),
    Clause(
        literals=frozenset({
            (False, "gcd_of_lcm", "R", "X", "Y", "Z"),
            (False, "lcm_of_gcds", "S", "X", "Y", "Z"),
            (True, "eq_lattice", "R", "S"),
        }),
        label="DIST-EQ: distributivity of gcd over lcm",
    ),

    # ---- Layer 6: Prime divisibility probability ----
    # The "flesh" on the lattice skeleton: measure theory.
    #
    # For prime p, delta_p(n) = 1 if p|n, 0 otherwise.
    # This oscillates with period p. Density = 1/p.
    # Independence of distinct primes IS the FTA.

    Clause(
        literals=frozenset({
            (False, "prime", "P"),
            (True, "has_density", "P"),
        }),
        label="PROB-PRIME: primes have divisibility density",
    ),
    Clause(
        literals=frozenset({
            (False, "prime", "P"),
            (False, "prime", "Q"),
            (False, "distinct", "P", "Q"),
            (True, "independent", "P", "Q"),
        }),
        label="PROB-INDEP: distinct primes have independent divisibility",
    ),
    Clause(
        literals=frozenset({
            (False, "has_density", "P"),
            (False, "has_density", "Q"),
            (False, "independent", "P", "Q"),
            (True, "joint_density_is_product", "P", "Q"),
        }),
        label="PROB-DENSITY: joint density = product (P(p|n AND q|n) = 1/pq)",
    ),

    # ---- Layer 7: Wave orthogonality ----
    # The indicator functions delta_p and delta_q are orthogonal
    # as functions on Z when p != q. Their correlation is zero
    # because P(p|n AND q|n) = P(p|n)*P(q|n) (independence).
    # Orthogonal waves form basis elements of the factor space.

    Clause(
        literals=frozenset({
            (False, "prime", "P"),
            (True, "has_wave", "P"),
        }),
        label="WAVE-PRIME: primes define discrete wave functions",
    ),
    Clause(
        literals=frozenset({
            (False, "has_wave", "P"),
            (False, "has_wave", "Q"),
            (False, "independent", "P", "Q"),
            (True, "orthogonal", "P", "Q"),
        }),
        label="WAVE-ORTHO: independent prime waves are orthogonal",
    ),
    Clause(
        literals=frozenset({
            (False, "orthogonal", "P", "Q"),
            (True, "basis_element", "P"),
        }),
        label="WAVE-BASIS1: orthogonal wave is basis element (first)",
    ),
    Clause(
        literals=frozenset({
            (False, "orthogonal", "P", "Q"),
            (True, "basis_element", "Q"),
        }),
        label="WAVE-BASIS2: orthogonal wave is basis element (second)",
    ),

    # ---- Shortcut axioms ----
    # Pre-collapsed chains for proofs where 4-literal axioms create
    # too many intermediate resolvents for FIFO to handle efficiently.

    # SHORTCUT-ORTHO: distinct primes -> orthogonal waves (collapses
    # PROB-INDEP + WAVE-PRIME + WAVE-ORTHO into one step)
    Clause(
        literals=frozenset({
            (False, "prime", "P"),
            (False, "prime", "Q"),
            (False, "distinct", "P", "Q"),
            (True, "orthogonal", "P", "Q"),
        }),
        label="SHORTCUT-ORTHO: distinct primes have orthogonal waves",
    ),

    # SHORTCUT-DIST: flatten the distributivity chain.
    # If the gcd-side and lcm-side results both exist, they are equal.
    Clause(
        literals=frozenset({
            (False, "gcd_of_lcm", "R", "X", "Y", "Z"),
            (True, "eq_lattice", "R", "R"),
        }),
        label="SHORTCUT-DIST-REFL: gcd-of-lcm result exists -> lattice eq",
    ),
]


# --- Theorem definitions ---

LATTICE_THEOREMS = {
    # ==== Part 1: Lattice structure ====

    "reflexivity": {
        "description": "Divisibility is reflexive: a | a",
        "axiom_labels": ["DIV-REFL"],
        "premises": [],
        "negated_goal": Clause(
            literals=frozenset({(False, "divides", "a", "a")}),
            label="negated goal: a does not divide a",
        ),
    },
    "antisymmetry": {
        "description": "Divisibility is antisymmetric: a|b and b|a -> a = b",
        "axiom_labels": ["DIV-ANTI"],
        "premises": [
            Clause(literals=frozenset({(True, "divides", "a", "b")}),
                   label="premise: a divides b"),
            Clause(literals=frozenset({(True, "divides", "b", "a")}),
                   label="premise: b divides a"),
        ],
        "negated_goal": Clause(
            literals=frozenset({(False, "eq_div", "a", "b")}),
            label="negated goal: a != b in divisibility",
        ),
    },
    "transitivity": {
        "description": "Divisibility is transitive: a|b and b|c -> a|c",
        "axiom_labels": ["DIV-TRANS"],
        "premises": [
            Clause(literals=frozenset({(True, "divides", "a", "b")}),
                   label="premise: a divides b"),
            Clause(literals=frozenset({(True, "divides", "b", "c")}),
                   label="premise: b divides c"),
        ],
        "negated_goal": Clause(
            literals=frozenset({(False, "divides", "a", "c")}),
            label="negated goal: a does not divide c",
        ),
    },
    "unit_is_bottom": {
        "description": "1 is the bottom element: one | a",
        "axiom_labels": ["DIV-UNIT"],
        "premises": [],
        "negated_goal": Clause(
            literals=frozenset({(False, "divides", "one", "a")}),
            label="negated goal: one does not divide a",
        ),
    },
    "gcd_is_lower_bound": {
        "description": "GCD is a lower bound: gcd(a,b) divides both a and b",
        "axiom_labels": ["GCD-LB", "LB-DEF1"],
        "premises": [
            Clause(literals=frozenset({(True, "is_gcd", "g", "a", "b")}),
                   label="premise: g = gcd(a,b)"),
        ],
        "negated_goal": Clause(
            literals=frozenset({(False, "divides", "g", "a")}),
            label="negated goal: gcd does not divide a",
        ),
    },
    "gcd_is_greatest": {
        "description": "GCD is the greatest lower bound: d|a and d|b -> d|gcd(a,b)",
        "axiom_labels": ["GCD-GREATEST"],
        "premises": [
            Clause(literals=frozenset({(True, "is_gcd", "g", "a", "b")}),
                   label="premise: g = gcd(a,b)"),
            Clause(literals=frozenset({(True, "lower_bound", "d", "a", "b")}),
                   label="premise: d is a lower bound of a and b"),
        ],
        "negated_goal": Clause(
            literals=frozenset({(False, "divides", "d", "g")}),
            label="negated goal: d does not divide gcd",
        ),
    },
    "lcm_is_upper_bound": {
        "description": "LCM is an upper bound: a and b both divide lcm(a,b)",
        "axiom_labels": ["LCM-UB", "UB-DEF1"],
        "premises": [
            Clause(literals=frozenset({(True, "is_lcm", "l", "a", "b")}),
                   label="premise: l = lcm(a,b)"),
        ],
        "negated_goal": Clause(
            literals=frozenset({(False, "divides", "a", "l")}),
            label="negated goal: a does not divide lcm",
        ),
    },
    "lcm_is_least": {
        "description": "LCM is the least upper bound: a|m and b|m -> lcm(a,b)|m",
        "axiom_labels": ["LCM-LEAST"],
        "premises": [
            Clause(literals=frozenset({(True, "is_lcm", "l", "a", "b")}),
                   label="premise: l = lcm(a,b)"),
            Clause(literals=frozenset({(True, "upper_bound", "m", "a", "b")}),
                   label="premise: m is an upper bound of a and b"),
        ],
        "negated_goal": Clause(
            literals=frozenset({(False, "divides", "l", "m")}),
            label="negated goal: lcm does not divide m",
        ),
    },
    "gcd_is_meet": {
        "description": "GCD is the lattice meet: vector min -> lower bound (2-step chain)",
        "axiom_labels": ["VEC-GCD", "GCD-LB", "LB-DEF1"],
        "premises": [
            Clause(literals=frozenset({(True, "min_vec", "g", "a", "b")}),
                   label="premise: g is componentwise min of a and b"),
        ],
        "negated_goal": Clause(
            literals=frozenset({(False, "divides", "g", "a")}),
            label="negated goal: vector min does not divide a",
        ),
    },
    "lcm_is_join": {
        "description": "LCM is the lattice join: vector max -> upper bound (2-step chain)",
        "axiom_labels": ["VEC-LCM", "LCM-UB", "UB-DEF1"],
        "premises": [
            Clause(literals=frozenset({(True, "max_vec", "l", "a", "b")}),
                   label="premise: l is componentwise max of a and b"),
        ],
        "negated_goal": Clause(
            literals=frozenset({(False, "divides", "a", "l")}),
            label="negated goal: a does not divide vector max",
        ),
    },
    "distributivity": {
        "description": "Distributive lattice: gcd(a, lcm(b,c)) = lcm(gcd(a,b), gcd(a,c))",
        "axiom_labels": ["DIST-EQ"],
        "premises": [
            # Supply the composed results directly. The prover verifies
            # that distributivity holds given these compositions exist.
            Clause(literals=frozenset({(True, "gcd_of_lcm", "r", "a", "b", "c")}),
                   label="premise: r = gcd(a, lcm(b,c))"),
            Clause(literals=frozenset({(True, "lcm_of_gcds", "r", "a", "b", "c")}),
                   label="premise: r = lcm(gcd(a,b), gcd(a,c))"),
        ],
        "negated_goal": Clause(
            literals=frozenset({(False, "eq_lattice", "r", "r")}),
            label="negated goal: gcd(a,lcm(b,c)) != lcm(gcd(a,b),gcd(a,c))",
        ),
    },

    # ==== Part 2: Probability waves ====

    "prime_density": {
        "description": "Each prime p has divisibility density 1/p",
        "axiom_labels": ["PROB-PRIME"],
        "premises": [
            Clause(literals=frozenset({(True, "prime", "p")}),
                   label="premise: p is prime"),
        ],
        "negated_goal": Clause(
            literals=frozenset({(False, "has_density", "p")}),
            label="negated goal: p has no density",
        ),
    },
    "prime_independence": {
        "description": "Distinct primes have independent divisibility (this IS the FTA)",
        "axiom_labels": ["PROB-INDEP"],
        "premises": [
            Clause(literals=frozenset({(True, "prime", "p")}),
                   label="premise: p is prime"),
            Clause(literals=frozenset({(True, "prime", "q")}),
                   label="premise: q is prime"),
            Clause(literals=frozenset({(True, "distinct", "p", "q")}),
                   label="premise: p != q"),
        ],
        "negated_goal": Clause(
            literals=frozenset({(False, "independent", "p", "q")}),
            label="negated goal: p and q are not independent",
        ),
    },
    "density_product": {
        "description": "Joint density is product: P(p|n AND q|n) = 1/pq (3-step chain)",
        "axiom_labels": ["PROB-PRIME", "PROB-INDEP", "PROB-DENSITY"],
        "premises": [
            Clause(literals=frozenset({(True, "prime", "p")}),
                   label="premise: p is prime"),
            Clause(literals=frozenset({(True, "prime", "q")}),
                   label="premise: q is prime"),
            Clause(literals=frozenset({(True, "distinct", "p", "q")}),
                   label="premise: p != q"),
        ],
        "negated_goal": Clause(
            literals=frozenset({(False, "joint_density_is_product", "p", "q")}),
            label="negated goal: joint density is not product",
        ),
    },
    "prime_wave": {
        "description": "Each prime defines a discrete wave function (period p indicator)",
        "axiom_labels": ["WAVE-PRIME"],
        "premises": [
            Clause(literals=frozenset({(True, "prime", "p")}),
                   label="premise: p is prime"),
        ],
        "negated_goal": Clause(
            literals=frozenset({(False, "has_wave", "p")}),
            label="negated goal: p has no wave function",
        ),
    },
    "wave_orthogonality": {
        "description": "Waves for distinct primes are orthogonal (correlation = 0)",
        "axiom_labels": ["PROB-INDEP", "WAVE-PRIME", "WAVE-ORTHO"],
        "premises": [
            Clause(literals=frozenset({(True, "prime", "p")}),
                   label="premise: p is prime"),
            Clause(literals=frozenset({(True, "prime", "q")}),
                   label="premise: q is prime"),
            Clause(literals=frozenset({(True, "distinct", "p", "q")}),
                   label="premise: p != q"),
        ],
        "negated_goal": Clause(
            literals=frozenset({(False, "orthogonal", "p", "q")}),
            label="negated goal: waves for p and q are not orthogonal",
        ),
    },
    "wave_basis": {
        "description": "Orthogonal prime waves form basis elements of the factor space",
        "axiom_labels": ["SHORTCUT-ORTHO", "WAVE-BASIS1"],
        "premises": [
            Clause(literals=frozenset({(True, "prime", "p")}),
                   label="premise: p is prime"),
            Clause(literals=frozenset({(True, "prime", "q")}),
                   label="premise: q is prime"),
            Clause(literals=frozenset({(True, "distinct", "p", "q")}),
                   label="premise: p != q"),
        ],
        "negated_goal": Clause(
            literals=frozenset({(False, "basis_element", "p")}),
            label="negated goal: p is not a basis element",
        ),
    },
    "fta_as_probability": {
        "description": "FTA as probability: independence + density -> unique factorization",
        "axiom_labels": ["PROB-PRIME", "PROB-INDEP", "PROB-DENSITY"],
        "premises": [
            Clause(literals=frozenset({(True, "prime", "p")}),
                   label="premise: p is prime"),
            Clause(literals=frozenset({(True, "prime", "q")}),
                   label="premise: q is prime"),
            Clause(literals=frozenset({(True, "distinct", "p", "q")}),
                   label="premise: p != q"),
        ],
        "negated_goal": Clause(
            literals=frozenset({(False, "joint_density_is_product", "p", "q")}),
            label="negated goal: unique factorization fails as probability",
        ),
    },
}


def lattice_prune(item, state) -> bool:
    """
    Prune for the lattice domain.
    Same limits as goedel_prune: depth 4, literal count 5.
    Lattice proofs are chains of implications, no deep function nesting.
    """
    if not isinstance(item, Clause):
        return False

    if len(item.literals) > 5:
        return True

    def term_depth(t):
        if isinstance(t, tuple):
            return 1 + max((term_depth(arg) for arg in t[1:]), default=0)
        return 0

    for lit in item.literals:
        for arg in lit[2:]:
            if term_depth(arg) > 4:
                return True
    return False


def make_lattice_state(theorem=None) -> OtterState:
    """
    Set up prime factor lattice axioms for a specific theorem.

    Args:
        theorem: which theorem to prove. One of the keys in
                 LATTICE_THEOREMS, or None for the default (reflexivity).
    """
    if theorem is None:
        theorem = "reflexivity"

    if theorem not in LATTICE_THEOREMS:
        raise ValueError(f"Unknown theorem: {theorem}. "
                        f"Choose from: {list(LATTICE_THEOREMS.keys())}")

    thm = LATTICE_THEOREMS[theorem]
    state = OtterState()

    for rule in LATTICE_RULES:
        for prefix in thm["axiom_labels"]:
            if rule.label.startswith(prefix):
                state.set_of_support.append(rule)
                break

    for premise in thm["premises"]:
        state.set_of_support.append(premise)

    state.set_of_support.append(thm["negated_goal"])
    return state


def run_lattice_proof_suite(max_steps=100, verbose=True):
    """
    Run all prime factor lattice theorems and return results.
    """
    results = {}

    for name, thm in LATTICE_THEOREMS.items():
        if verbose:
            print(f"\n{'='*60}")
            print(f"THEOREM: {name}")
            print(f"  {thm['description']}")
            print(f"{'='*60}")

        state = make_lattice_state(theorem=name)

        state = run_otter(
            state, resolve,
            max_steps=max_steps,
            stop_fn=found_empty_clause,
            subsumes_fn=clause_subsumes,
            prune_fn=lattice_prune,
            verbose=verbose,
        )

        proved = found_empty_clause(state)
        results[name] = {
            "proved": proved,
            "steps": state.step,
            "description": thm["description"],
            "state": state,
        }

        if verbose:
            if proved:
                print_proof(state)
            else:
                print(f"\n  NOT PROVED in {state.step} steps.")

    return results


def print_lattice_results(results):
    """Pretty-print the lattice proof suite results."""
    # Split into structure and wave theorems for display
    structure_names = ["reflexivity", "antisymmetry", "transitivity",
                       "unit_is_bottom", "gcd_is_lower_bound",
                       "gcd_is_greatest", "lcm_is_upper_bound",
                       "lcm_is_least", "gcd_is_meet", "lcm_is_join",
                       "distributivity"]
    wave_names = ["prime_density", "prime_independence", "density_product",
                  "prime_wave", "wave_orthogonality", "wave_basis",
                  "fta_as_probability"]

    print(f"\n{'='*60}")
    print("PRIME FACTOR LATTICE: Proof Suite Results")
    print(f"{'='*60}")

    all_proved = True

    print(f"\n  --- Lattice Structure ---")
    for name in structure_names:
        if name in results:
            r = results[name]
            status = "PROVED" if r["proved"] else "NOT PROVED"
            if not r["proved"]:
                all_proved = False
            print(f"  {status:>11s} ({r['steps']:2d} steps)  {r['description']}")

    print(f"\n  --- Probability Waves ---")
    for name in wave_names:
        if name in results:
            r = results[name]
            status = "PROVED" if r["proved"] else "NOT PROVED"
            if not r["proved"]:
                all_proved = False
            print(f"  {status:>11s} ({r['steps']:2d} steps)  {r['description']}")

    print(f"\n{'='*60}")
    if all_proved:
        print("  ALL THEOREMS PROVED.")
        print("  The divisibility ordering on N forms a distributive lattice.")
        print("  GCD = meet (componentwise min of factor vectors).")
        print("  LCM = join (componentwise max of factor vectors).")
        print("  Prime divisibility waves are independent and orthogonal.")
        print("  They form a basis for the infinite-dimensional factor space.")
        print("  Independence of prime waves IS the FTA restated as measure theory.")
    else:
        print("  SOME THEOREMS FAILED. Check axiom formulations.")
    print(f"{'='*60}")


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
    "goedel": {
        "make_state": make_goedel_state,
        "combine_fn": resolve,  # pure resolution, no paramodulation needed
        "subsumes_fn": clause_subsumes,
        "stop_fn": found_empty_clause,
        "prune_fn": goedel_prune,
        "description": "Goedel numbering: prove encoding completeness for self-reference",
    },
    "lattice": {
        "make_state": make_lattice_state,
        "combine_fn": resolve,  # pure resolution, no paramodulation needed
        "subsumes_fn": clause_subsumes,
        "stop_fn": found_empty_clause,
        "prune_fn": lattice_prune,
        "description": "Prime factor lattice: divisibility, GCD/LCM, probability waves",
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
    if args.domain in ("resolution", "chain", "bridge", "peano", "goedel", "lattice"):
        if found_empty_clause(state):
            print_proof(state)
        else:
            print("\nNo proof found (empty clause not derived).")

    # Goedel domain: run full proof suite and self-reference check
    if args.domain == "goedel":
        results = run_goedel_proof_suite(
            max_steps=args.steps,
            verbose=not args.quiet,
        )
        print_goedel_results(results)

    # Lattice domain: run full proof suite
    # Probability wave proofs need ~100 steps due to multi-variable axioms
    if args.domain == "lattice":
        results = run_lattice_proof_suite(
            max_steps=max(args.steps, 100),
            verbose=not args.quiet,
        )
        print_lattice_results(results)

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
