# CLAUDE.md

## What this is

A generalization of the Otter theorem prover's main loop as a pluggable
combinatorial search algorithm, extended with edge-first knowledge graphs
and a causal encoding that turns cause-and-effect into number theory.

Based on Johnicholas Hines' formulation of the Otter main loop.
Extended by Hallie Larsson with edge-first knowledge graph concepts.

## The core idea (for newcomers)

Give every event in a causal chain its own prime number. Then define
each event's code as its prime times the codes of everything that caused it:

    gn(E) = p_E × ∏ gn(causes of E)

Now arithmetic *is* causation:

- **A caused B?** Check if gn(A) divides gn(B). That's it.
- **Shared history?** Compute gcd(gn(A), gn(B)). The prime factors
  are exactly the common ancestors.
- **Combined future?** Compute lcm(gn(A), gn(B)). It encodes the
  minimal event downstream of both.

This is not a metaphor. It is the Fundamental Theorem of Arithmetic
applied to causal structure. Every DAG gets a unique, injective encoding.
The divisibility lattice of the natural numbers becomes a calculus of
causation. See `domains/causal_encoding.py`, Properties 6 and 8.

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
python -m pytest                         # 349 tests, ~6s
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

## Reading order

1. `core/engine.py` (157 lines) — the Otter main loop. Start here.
2. `core/state.py` (230 lines) — Item, Edge, Clause, OtterState.
3. `conditional_proof.py` (191 lines) — the ex falso guard.
4. A domain of your choice — `domains/little_alchemy.py` is the simplest.
5. `domains/causal_encoding.py` — the causal encoding (the core idea above).
6. `causal_calculus.py` — convergent proofs, zeta connection, self-reference.

## Architecture

- **Core loop**: `otter_step()` / `run_otter()` in `core/engine.py`
- **Three item types**: `Item` (classic), `Edge` (relationship-first), `Clause` (symbolic logic)
- **Bone and flesh**: Clause is rigid symbolic bone, Edge is probabilistic flesh.
  `core/bridge.py` connects them. `stiffen_edges()` raises proven edges to confidence 1.0.
- **Conditional proofs**: `conditional_proof.py` guards against ex falso and confidence laundering.
- **Convergent proofs**: `causal_calculus.py` tracks confidence as evidence accumulates.
- **Causal encoding**: `domains/causal_encoding.py` encodes causal DAGs into the
  natural numbers so that divisibility = causality. Then explores what happens
  when you put complex amplitudes on this structure.

## Domains

Nine domains live in the `DOMAINS` registry (`domains/__init__.py`) and run the
standard Otter loop: little_alchemy, edges, resolution, chain, bridge, peano,
goedel, lattice, interactive.

Four domains are special-cased in `__main__.py` with their own control flow:
zeta, self-ref, causal_encoding (each bypass the Otter loop), and llm (uses
Claude API as combine_fn).

## What is established and what is not

The causal encoding (divisibility = causality) is proved by construction.
It is the diamond-hard core of the project. The diamond-hard properties:
exponents count paths (P6), inner products count correlated path pairs (P7),
the cone angle converges to √((k-1)/k) (P8), and the Gram matrix is
invariant under any prime assignment (P9). All proved, no gaps.

On top of the encoding, `causal_encoding.py` explores what happens when you
assign complex amplitudes (Euler factors) and ask about probabilities and
interference. The Euler factor amplitudes and Born probabilities change
between encodings; the causal geometry does not. This is where the gaps live:

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
