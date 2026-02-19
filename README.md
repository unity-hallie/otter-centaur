# otter-centaur

A generalization of the [Otter theorem prover](https://www.cs.unm.edu/~mccune/otter/)'s
main loop as a pluggable combinatorial search algorithm, extended with edge-first
knowledge graphs and a causal encoding that turns cause-and-effect into number theory.

Based on Johnicholas Hines' formulation of the Otter main loop.
Extended by Hallie Larsson.

## The one idea

Give every event in a causal chain its own prime. Define each event's code as
its prime times the codes of everything that caused it:

```
gn(E) = p_E × ∏ gn(causes of E)
```

Then:

- **A caused B?** Check if `gn(A)` divides `gn(B)`.
- **Shared history?** `gcd(gn(A), gn(B))` — the common ancestors, exactly.
- **Downstream join?** `lcm(gn(A), gn(B))`.

This is the Fundamental Theorem of Arithmetic applied to causal structure.

Some things that fall out of it, which we did not go looking for:

- The exponent of `p_A` in `gn(E)` equals the number of directed paths from A to E. (Property 6)
- The inner product of two event vectors counts correlated path pairs through shared ancestors. (Property 7)
- For k-ary nested diamonds, the causal overlap converges to `√((k-1)/k)` — a light cone angle that depends only on the branching factor. (Property 8)
- The prime assignment is a **gauge symmetry**: the causal geometry is invariant under any relabeling of primes; the wave amplitudes are not. The gauge group acts freely. (Property 9)

Properties 6–9 are proved. No gaps.

## What's not settled

The code is honest about this. From `causal_encoding.py`:

- **Gap 1** (partial): The Euler factor `1/(1-p^{-s})` is forced by multiplicativity and self-similarity. Why self-similar, and why `σ = 1/2`, remain open.
- **Gap 2** (partial): Path interference produces genuine cross-terms, but the path-sum doesn't equal the per-event product amplitude.
- **Gap 3** (partial): Gleason's theorem applies to the Hilbert space. The bridge to the specific Born normalization is not established.
- **Gap 4** (open): `t` is a parameter. There is no Hamiltonian.

## Where to look

```
otter/domains/causal_encoding.py   # the core idea and Properties 6-9
otter/core/engine.py               # the Otter loop (157 lines)
otter/causal_calculus.py           # convergent proofs, zeta, self-reference
tests/test_causal_encoding.py      # 84 tests including gauge symmetry
```

Full reading order and architecture in [CLAUDE.md](CLAUDE.md).

## Running

```
pip install -e .
python -m otter --domain causal_encoding
python -m otter --domain peano
python -m otter --domain goedel
python -m pytest                   # 351 tests, ~6s
```

## License

MIT with ethical notice — no military, weapons, or surveillance use.
See [LICENSE](LICENSE).
