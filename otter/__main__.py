"""
CLI entry point. Run as: python -m otter --domain <name>
"""

import argparse
import os
import sys

from .core.state import OtterState
from .core.engine import run_otter
from .core.proof import found_empty_clause, print_proof
from .core.bridge import stiffen_edges
from .visualization import print_state, print_history, export_dot
from .domains import DOMAINS
from .domains.llm import make_llm_combine
from .domains.little_alchemy import make_little_alchemy_state
from .domains.goedel import run_goedel_proof_suite, print_goedel_results
from .domains.lattice import run_lattice_proof_suite, print_lattice_results
from .causal_calculus import run_zeta_approach, run_self_referential_approach
from .domains.causal_encoding import run_hilbert_space_demo


def main():
    parser = argparse.ArgumentParser(description="Otter combinatorial search")
    parser.add_argument(
        "--domain",
        choices=list(DOMAINS.keys()) + ["llm", "zeta", "self-ref", "causal_encoding"],
        default="little_alchemy",
        help="Which domain to explore",
    )
    parser.add_argument("--zeta-s", type=float, default=2.0,
                        help="Exponent s for zeta domain (default 2.0; must be > 1)")
    parser.add_argument("--steps",  type=int, default=20,   help="Max steps")
    parser.add_argument("--save",   type=str, default=None, help="Save state to file")
    parser.add_argument("--load",   type=str, default=None, help="Load state from file")
    parser.add_argument("--dot",    type=str, default=None, help="Export DOT graph to file")
    parser.add_argument("--quiet",  action="store_true",    help="Less output")
    args = parser.parse_args()

    # Zeta domain is handled entirely separately -- it's a convergence
    # demonstration, not an Otter loop run.
    if args.domain == "zeta":
        run_zeta_approach(s=args.zeta_s, num_primes=15, verbose=not args.quiet)
        return

    # Self-referential domain: the proof that operates on itself.
    if args.domain == "self-ref":
        run_self_referential_approach(s=args.zeta_s, verbose=not args.quiet)
        return

    # Causal encoding domain: Hilbert space demo, not an Otter loop run.
    if args.domain == "causal_encoding":
        run_hilbert_space_demo(verbose=not args.quiet)
        return

    # --- Load or build initial state ---
    if args.load:
        state = OtterState.load(args.load)
        print(f"Loaded state from {args.load} (step {state.step})")
    elif args.domain == "llm":
        state = make_little_alchemy_state()
    else:
        state = DOMAINS[args.domain]["make_state"]()

    # --- Build combine_fn and kwargs ---
    if args.domain == "llm":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        combine_fn = make_llm_combine(api_key)
        kwargs = {}
        stop_fn = None
    else:
        domain = DOMAINS[args.domain]
        combine_fn = domain["combine_fn"]
        stop_fn = domain.get("stop_fn")
        kwargs = {}
        for key in ("choose_focus_fn", "subsumes_fn", "prune_fn"):
            if key in domain:
                kwargs[key] = domain[key]

    print(f"Domain: {args.domain}")
    print_state(state)

    # --- Run ---
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

    # --- Post-processing ---

    # Resolution domains: show proof if found
    if args.domain in ("resolution", "chain", "bridge", "peano", "goedel", "lattice"):
        if found_empty_clause(state):
            print_proof(state)
        else:
            print("\nNo proof found (empty clause not derived).")

    # Goedel: run full proof suite
    if args.domain == "goedel":
        results = run_goedel_proof_suite(
            max_steps=args.steps,
            verbose=not args.quiet,
        )
        print_goedel_results(results)

    # Lattice: run full proof suite (wave proofs need ~100 steps)
    if args.domain == "lattice":
        results = run_lattice_proof_suite(
            max_steps=max(args.steps, 100),
            verbose=not args.quiet,
        )
        print_lattice_results(results)

    # Bridge: demonstrate edge stiffening
    if args.domain == "bridge" and found_empty_clause(state):
        uncertain = getattr(state, "_uncertain_edges", None)
        if uncertain:
            from .core.state import Clause
            all_clauses = [
                c for c in list(state.set_of_support) + state.usable
                if isinstance(c, Clause)
            ]
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


if __name__ == "__main__":
    main()
