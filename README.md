# otter-centaur

A pluggable generalization of the [Otter theorem prover](https://www.cs.unm.edu/~mccune/otter/) main loop, extended with edge-first knowledge graphs and a causal encoding.

Based on Johnicholas Hines' formulation of the Otter main loop. Extended by Hallie Larsson.

---

Assign each event in a causal DAG a fresh prime. Define its code recursively:

```
gn(E) = p_E × ∏ gn(causes of E)
```

Causality becomes divisibility. GCD gives shared ancestry. LCM gives causal join. Some other things fall out — path counting, inner products, cone angles, a gauge symmetry — that we weren't looking for. They're in `otter/domains/causal_encoding.py` with proofs and the gaps named.

```
pip install -e .
python -m otter --domain causal_encoding
python -m pytest   # 351 tests
```

See [CLAUDE.md](CLAUDE.md) for architecture and reading order.

MIT with ethical notice — no military, weapons, or surveillance use.
