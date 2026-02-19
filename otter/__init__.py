"""
Otter: combinatorial search engine inspired by the Otter theorem prover.

Based on Johnicholas Hines' formulation of the Otter main loop as a
generalized algorithm. Extended with edge-first knowledge graphs.

Usage:
    python -m otter --domain little_alchemy
    python -m otter --domain edges
    python -m otter --domain peano
    python -m otter --domain goedel
    python -m otter --domain lattice
    python -m otter --domain interactive
    python -m otter --domain llm       (needs ANTHROPIC_API_KEY)
"""

from .core.state import Item, Edge, Clause, OtterState
from .core.engine import otter_step, run_otter
from .core.proof import found_empty_clause, extract_proof, print_proof
from .core.bridge import clause_from_edge, edge_from_clause, stiffen_edges
from .inference.resolve import resolve, clause_subsumes
from .inference.paramodulate import paramodulate, resolve_and_paramodulate
from .conditional_proof import ConditionalProof, prove_conditionally

__all__ = [
    "Item", "Edge", "Clause", "OtterState",
    "otter_step", "run_otter",
    "found_empty_clause", "extract_proof", "print_proof",
    "clause_from_edge", "edge_from_clause", "stiffen_edges",
    "resolve", "clause_subsumes",
    "paramodulate", "resolve_and_paramodulate",
    "ConditionalProof", "prove_conditionally",
]
