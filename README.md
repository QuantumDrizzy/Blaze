# Blaze — Tensor-Network compression of high-order data

Lossy TT/MPS compression + approximate reconstruction for data with **genuine
tensor-train structure** — primarily **quantum states** (Cirq) and TT-native
scientific tensors — with honest compressibility diagnostics and a load-bearing
quantum layer (the MPS ↔ quantum-circuit correspondence).

**Honest scope (the Phase-1 fair benchmark settled this):** TT only *beats a fair
matrix SVD* on **genuinely TT-native data** — low rank across *every* cut, structure
spread over modes (a low-bond MPS; a **low-entanglement quantum state**, guaranteed by
physics). On generic "structured" data (smooth fields, hyperspectral-like volumes) a
well-chosen single matrix SVD competes at **equal parameters** — "high-order" or
"smooth" is **not** enough; generic high-entropy data doesn't compress at all
(`[KNOWN_LIMIT]`). **Blaze is not a general compressor.** Its real, usable value:
(1) **quantum-state compression + fidelity/sampling**, (2) **honest diagnostics** that
tell you *whether* TT fits your data, (3) fast TT where it does. Lossy; compared
against the best matrix SVD at matched params, never against lossless Zstd.

Architecture, build sequence, and the honesty contract:
[`docs/ADR-0001`](docs/ADR-0001-blaze-architecture.md).

## Stack (each language where it fits)

`Rust` ingestion · CLI · persistence  ·  `C++/CUDA` TT kernels (hot core)  ·
`Python` TT algorithms + benchmarks  ·  `Cirq` MPS↔circuit bridge

## Status

**Phase 0 — architecture defined.** Implementation pending. Co-developed: spec +
review/benchmark-gate by Claude, generation by Grok. Nothing ships that fails the
honesty gates in the ADR (right baselines, documented limits, no overclaim).

**Fase 1 (current)**: Pure-Python executable spec + golden tests + **the honest
classical benchmark that decides claim (b)**.

- `blaze.compress` + `TT` class with the final API shape (will be 1:1 with Rust).
- Full Cirq integration (`blaze.cirq`) — GHZ, product states, shallow circuits as
  regression gates (verifies impl correctness via physics).
- `blaze.diagnostics.analyze_compressibility` + singular decay — makes
  "is this data even TT-compressible?" measurable (effective ranks, entanglement-spectrum analogue).
- `python/blaze/examples/classical_benchmark.py`: the real test (hyperspectral-like,
  smooth fields, image stacks vs random + matrix-SVD baseline). This is what you run
  to see if Blaze is real before writing one line of CUDA.
- No custom .blz format yet (export cores as .npy or stay in-memory). dtypes: f32/f64/c128.
  c32 deferred. cudarc planned for CUDA phase.

Run the gate that matters:

```bash
# after `pip install -e '.[quantum]'` (or without quantum for classical only)
python -m blaze.examples.classical_benchmark
python -m blaze.examples.quantum_compression   # needs cirq
```

See also: `blaze --bench`, `blaze --quantum` (after install).

