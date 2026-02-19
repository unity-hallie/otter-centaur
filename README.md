# otter-centaur

A pluggable generalization of the [Otter theorem prover](https://www.cs.unm.edu/~mccune/otter/) main loop, extended with edge-first knowledge graphs and a causal encoding.

Based on Johnicholas Hines' formulation of the Otter main loop. Extended by Hallie Larsson.

---

Assign each event in a causal DAG a fresh prime. Define its code recursively:

```
gn(E) = p_E × ∏ gn(causes of E)
```

Causality becomes divisibility. GCD gives shared ancestry. LCM gives causal join. Some other things fall out — path counting, inner products, cone angles, a gauge symmetry — that we weren't looking for. They're in `otter/domains/causal_encoding.py` with proofs and the gaps named.

---

## Ways to use it

**As a human-AI collaboration engine.**
The core loop (`core/engine.py`, 157 lines) is just: pick a focus, combine it with everything known, add results back. The combination function is a parameter. Swap in a human, an LLM, or both:

```python
python -m otter --domain interactive   # you decide what combines with what
python -m otter --domain llm           # Claude decides
# or write a centaur: human picks focus, LLM proposes combinations, human approves
```

The loop doesn't care which. That's the point.

**As a shared representation for causal reasoning.**
If two parties (human, AI, or both) are reasoning about cause and effect, the encoding gives them a common language: divisibility is causality, primes are events, GCD is shared history. No proprietary format, just arithmetic.

**As a substrate for honest knowledge graphs.**
Items are edges first, nodes second. Every claim carries a confidence. `conditional_proof.py` refuses to derive conclusions from contradictions. The gaps are documented in the code, not hidden.

**As a starting point for something else.**
The loop is generic. The domains show what it can do: theorem proving, Peano arithmetic, Gödel numbering, prime lattices, convergent proofs. The combination function is yours to define.

---

```
pip install -e .
python -m otter --domain causal_encoding
python -m pytest   # 351 tests
```

See [CLAUDE.md](CLAUDE.md) for architecture and reading order.

MIT with ethical notice — no military, weapons, or surveillance use.
