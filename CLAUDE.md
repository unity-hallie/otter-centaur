# CLAUDE.md

## What this is

A generalization of the Otter theorem prover's main loop as a pluggable
combinatorial search algorithm, extended with edge-first knowledge graphs,
conditional proofs, and a causal encoding that connects prime factorization
to causal structure.

Based on Johnicholas Hines' formulation of the Otter main loop.
Extended by Hallie Larsson with edge-first knowledge graph concepts.

## Running

```
python -m otter                          # default: little_alchemy
python -m otter --domain resolution      # first-order logic
python -m otter --domain peano           # proves 1+1=2
python -m otter --domain goedel          # Gödel numbering proofs
python -m otter --domain lattice         # prime factor lattice
python -m otter --domain zeta            # Euler product convergence
python -m otter --domain self-ref        # self-referential fixed point
python -m otter --domain causal_encoding # Hilbert space + interference
python -m otter --domain interactive     # human in the loop
python -m pytest                         # 324 tests, ~6s
```

## Design principles (SPICES)

Discoverable: `grep -r "SPICES" otter/`

- **Simplicity** — `core/engine.py`: the Otter loop stripped to its essence.
  Pick focus, combine with usable, add results back. Nothing more.

- **Peace** — `conditional_proof.py`: the ex falso guard. The system refuses
  to derive conclusions from contradictions. Confidence returns 0.0.

- **Integrity** — `domains/causal_encoding.py`: say what is proved, name
  what is not. Every gap is documented. The code is its own testimony.

- **Community** — `core/engine.py`: the pluggable combine_fn. Human,
  LLM, deterministic, or centaur (human+LLM). The loop invites
  collaboration by design.

- **Equality** — `core/state.py`: Edge is the primary object, not
  subordinate to Clause or Item. Relationships first, entities second.
  Inspired by polysynthetic language structure.

- **Stewardship** — `causal_calculus.py`: do not claim more certainty
  than the evidence supports. Monotone tracking, Cauchy convergence
  certificates, honest limit estimation.

## Reading order (for newcomers)

1. `core/engine.py` (157 lines) — the Otter main loop. Start here.
2. `core/state.py` (230 lines) — Item, Edge, Clause, OtterState.
3. `conditional_proof.py` (191 lines) — the ex falso guard.
4. A domain of your choice — `domains/little_alchemy.py` is the simplest.
5. `causal_calculus.py` — convergent proofs, zeta connection, self-reference.
6. `domains/causal_encoding.py` — Hilbert space, interference, Born rule.

## Architecture

- **Core loop**: `otter_step()` / `run_otter()` in `core/engine.py`
- **Three item types**: `Item` (classic), `Edge` (relationship-first), `Clause` (symbolic logic)
- **Bone and flesh**: Clause is rigid symbolic bone, Edge is probabilistic flesh.
  `core/bridge.py` connects them. `stiffen_edges()` raises proven edges to confidence 1.0.
- **Conditional proofs**: `conditional_proof.py` guards against ex falso and confidence laundering.
- **Convergent proofs**: `causal_calculus.py` tracks confidence as evidence accumulates.
- **Causal encoding**: `domains/causal_encoding.py` constructs a Hilbert space from prime
  factorization vectors, demonstrates path interference, and invokes Gleason's theorem
  for the Born rule.

## Domains

Nine domains live in the `DOMAINS` registry (`domains/__init__.py`) and run the
standard Otter loop: little_alchemy, edges, resolution, chain, bridge, peano,
goedel, lattice, interactive.

Four domains are special-cased in `__main__.py` with their own control flow:
zeta, self-ref, causal_encoding (each bypass the Otter loop), and llm (uses
Claude API as combine_fn).

## Open gaps (honest accounting)

The causal encoding constructs a Hilbert space from prime factorization
vectors. Four gaps were identified. Two are partially closed, two remain open:

- **Gap 1** (open): The Euler factor amplitude is defined, not derived from
  causal structure. Why this specific complex function?
- **Gap 2** (partial): Path interference. The path-sum decomposition produces
  genuine complex cross-terms. But the path-sum is not equal to the per-event
  amplitude (product form) used by `born_probabilities()`.
- **Gap 3** (partial): Born rule. The prime exponent vector space is a Hilbert
  space (by FTA). For dimension >= 3, Gleason's theorem applies. But the
  bridge from Gleason's trace formula on projectors to the specific
  normalization `|ψ(E)|²/Σ|ψ(F)|²` is not established — event vectors
  are non-orthonormal.
- **Gap 4** (open): The parameter t is not physical time. There is no
  Hamiltonian or dynamics. Making t physical requires Berry-Keating
  or an equivalent construction.

## The stable axiom

The ethical notice in LICENSE is the only self-referential axiom in the system.
The Gödel numbering domain can encode all axioms, so if the notice were removed,
its absence would be a derivable fact. The proof lives in
`GOEDEL_THEOREMS["stable_axiom"]`. Every other axiom is contingent. This one
witnesses itself.

## License

MIT with ethical notice (no military/weapons/surveillance use).
See LICENSE.
