"""
Core data structures: Item, Edge, Clause, OtterState.

These are the atoms of the whole system. Nothing in here depends on
inference rules, domains, or combination strategies.

Terms and literals (for Clause):
    Variables:  strings starting with uppercase -- "X", "Y", "Foo"
    Constants:  strings starting with lowercase -- "0", "alice"
    Functions:  tuples  -- ("s", "0"), ("plus", "X", "Y")

    Literals:   (sign: bool, predicate: str, arg1, arg2, ...)
                (True, "human", "socrates")  ->  human(socrates)
                (False, "mortal", "X")       -> ~mortal(X)

    A Clause is a frozenset of literals (disjunction).
    The empty clause [] is a contradiction -> proof found.
"""

from dataclasses import dataclass, field
from typing import Optional
from collections import deque
import json


@dataclass
class Item:
    """An item in the Otter loop. The base type for classic Otter."""
    name: str
    content: str
    source: tuple = ()
    step: int = 0

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, Item) and self.name == other.name

    def __repr__(self):
        return f"Item({self.name!r})"


@dataclass
class Edge:
    """
    A relationship in an edge-first knowledge graph.

    In edge-first graphs the edge IS the primary object.
    Nodes are just where edges meet; they have no independent existence.

    SPICES: Equality — relationships are primary, not entities.
    """
    subject: str
    predicate: str
    object: str
    confidence: float = 0.7
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
        return {self.subject, self.object}

    def shares_term_with(self, other: 'Edge') -> set:
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


@dataclass
class Clause:
    """
    A disjunction of literals. The item type for resolution domains.

    The empty clause (literals = frozenset()) is contradiction / proof found.
    The bone in the bone-and-flesh metaphor: symbolically certain, no hedging.
    """
    literals: frozenset
    source: tuple = ()
    step: int = 0
    label: str = ""

    @property
    def name(self):
        if not self.literals:
            return "[]"
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


@dataclass
class OtterState:
    """
    Full state of the Otter loop, serializable for continuity.

    set_of_support: items that haven't been focused on yet (the frontier)
    usable:         items that have been focused on (exhausted)
    history:        log of what happened at each step
    """
    set_of_support: deque = field(default_factory=deque)
    usable: list = field(default_factory=list)
    history: list = field(default_factory=list)
    step: int = 0
    halted: bool = False
    halt_reason: str = ""

    def to_dict(self):
        def serialize_term(t):
            if isinstance(t, tuple):
                return {"_fn": [serialize_term(x) for x in t]}
            if isinstance(t, bool):
                return {"_bool": t}
            return t

        def serialize_literal(lit):
            return [serialize_term(t) for t in lit]

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
