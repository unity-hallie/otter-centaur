"""
Domain registry.

Each domain is a dict describing how to configure the Otter loop:
    make_state:       () -> OtterState
    combine_fn:       (x, y) -> list
    subsumes_fn:      (a, b) -> bool        [optional]
    prune_fn:         (item, state) -> bool [optional]
    stop_fn:          (state) -> bool       [optional]
    choose_focus_fn:  (sos) -> item         [optional]
    description:      str
"""

from .little_alchemy import make_little_alchemy_state, little_alchemy_combine
from .edges import make_edge_state, edge_combine, edge_subsumes
from .resolution import make_resolution_state, make_chain_resolution_state
from .bridge_demo import make_bridge_demo_state
from .peano import make_peano_state, peano_prune
from .goedel import make_goedel_state, goedel_prune
from .lattice import make_lattice_state, lattice_prune
from .interactive import interactive_combine, interactive_choose_focus

from ..inference.resolve import resolve, clause_subsumes
from ..inference.paramodulate import resolve_and_paramodulate
from ..core.proof import found_empty_clause


DOMAINS = {
    "little_alchemy": {
        "make_state":  make_little_alchemy_state,
        "combine_fn":  little_alchemy_combine,
        "description": "Classic Little Alchemy: combine elements to discover new ones",
    },
    "edges": {
        "make_state":  make_edge_state,
        "combine_fn":  edge_combine,
        "subsumes_fn": edge_subsumes,
        "description": "Edge-first knowledge graph: relationships combine via shared terms",
    },
    "resolution": {
        "make_state":  make_resolution_state,
        "combine_fn":  resolve,
        "subsumes_fn": clause_subsumes,
        "stop_fn":     found_empty_clause,
        "description": "Symbolic resolution: prove theorems via refutation",
    },
    "chain": {
        "make_state":  make_chain_resolution_state,
        "combine_fn":  resolve,
        "subsumes_fn": clause_subsumes,
        "stop_fn":     found_empty_clause,
        "description": "Multi-step resolution: prove a chain of implications",
    },
    "bridge": {
        "make_state":  make_bridge_demo_state,
        "combine_fn":  resolve,
        "subsumes_fn": clause_subsumes,
        "stop_fn":     found_empty_clause,
        "description": "Bone-flesh bridge: prove symbolic facts, stiffen uncertain edges",
    },
    "peano": {
        "make_state":  make_peano_state,
        "combine_fn":  resolve_and_paramodulate,
        "subsumes_fn": clause_subsumes,
        "stop_fn":     found_empty_clause,
        "prune_fn":    peano_prune,
        "description": "Peano arithmetic: prove 1+1=2 from first principles",
    },
    "goedel": {
        "make_state":  make_goedel_state,
        "combine_fn":  resolve,
        "subsumes_fn": clause_subsumes,
        "stop_fn":     found_empty_clause,
        "prune_fn":    goedel_prune,
        "description": "Goedel numbering: prove encoding completeness for self-reference",
    },
    "lattice": {
        "make_state":  make_lattice_state,
        "combine_fn":  resolve,
        "subsumes_fn": clause_subsumes,
        "stop_fn":     found_empty_clause,
        "prune_fn":    lattice_prune,
        "description": "Prime factor lattice: divisibility, GCD/LCM, probability waves",
    },
    "interactive": {
        "make_state":       make_little_alchemy_state,
        "combine_fn":       interactive_combine,
        "choose_focus_fn":  interactive_choose_focus,
        "description":      "Human in the loop: you choose focus and decide combinations",
    },
}
