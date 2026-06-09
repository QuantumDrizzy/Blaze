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
tell you *whether* TT fits your data, (3) fast TT where it does, (4) **operations in
the compressed representation** — inner product, fidelity, distance and exact
similarity search (`TTIndex`) computed on the TT directly, **without decompressing**
(Phase 7, `O(nχ³)` vs `O(2ⁿ)`; see [`docs/PHASE7-results.md`](docs/PHASE7-results.md)),
(5) **a second, composable quantization stage** — int8/4-bit core codes on top of the
TT rank truncation, error measured and composed (Phase 8: TFIM paramagnet 77× → **447×**
at fidelity 0.9999; see [`docs/PHASE8-results.md`](docs/PHASE8-results.md)).
Lossy; compared against the best matrix SVD at matched params, never against lossless Zstd.

Architecture, build sequence, and the honesty contract:
[`docs/ADR-0001`](docs/ADR-0001-blaze-architecture.md).

## Stack (each language where it fits)

`Rust` ingestion · CLI · persistence  ·  `C++/CUDA` TT kernels (hot core)  ·
`Python` TT algorithms + benchmarks  ·  `Cirq` MPS↔circuit bridge

## Status — 8 phases complete (measured on RTX 5060 Ti, sm_120)

Every number below is from a reproducible gate in `tests/` and written up in `docs/`.
Honest baselines, documented `[KNOWN_LIMIT]`s, no overclaim — the contract of
[`docs/ADR-0001`](docs/ADR-0001-blaze-architecture.md) and
[`docs/ADR-0002`](docs/ADR-0002-overlap-and-quantization.md).

| # | Phase | Headline result (measured) | Detail |
|--:|-------|----------------------------|--------|
| 1 | Honest classical benchmark | TT beats a *fair* matrix SVD **only on TT-native data**; generic "structured"/high-entropy data does not compress (`[KNOWN_LIMIT]`, documented not hidden) | [PHASE2 §0](docs/PHASE2-rust-core-spec.md) |
| 2 | Rust core `blaze-core` | 1:1 **parity** with the Python reference (CPU, faer/nalgebra); same ranks, `rel_error` within fp | [lib.rs](crates/blaze-core/src/lib.rs) |
| 3 | CUDA SVD offload (cuSOLVER) | **~2.9× (f64) / ~3.8× (c64)** above the **n≈24 crossover**; below it the CPU wins (transfer/launch-bound — stated, not hidden) | [PHASE3](docs/PHASE3-results.md) |
| 4 | Approximate reconstruction | **monotone** error↔rank dial; `.blz` roundtrip **75×** on a TFIM paramagnet at fidelity **1.000000000000** | [PHASE4](docs/PHASE4-results.md) |
| 5 | MPS → circuit synthesis | sequential state-prep (Schön 2005) reproduces the MPS at **fidelity 1.0** (GHZ, product, physical TFIM); ancilla disentangles | [PHASE5](docs/PHASE5-results.md) |
| 6 | Rust CLI + `.blz` persistence | `blaze compress` / `reconstruct` end-to-end, c64 verified on disk | [blz-format](docs/blz-format.md) |
| 7 | Ops in compressed space | `inner`/`fidelity`/`distance`/`TTIndex` via the MPS zipper, **exact to 1e-15**, `O(nχ³)`; runs at **n=40** (dense 2⁴⁰ = 17.6 TB, impossible) in **3.6 ms**; TFIM search recovers the **quantum phase transition** | [PHASE7](docs/PHASE7-results.md) |
| 8 | Core quantization | int8/4-bit codes on top of TT, error **composed** and measured; TFIM paramagnet **76.6× → 447×** at fidelity **0.99994** (int8), up to **705×** at 4-bit | [PHASE8](docs/PHASE8-results.md) |

**Flagship validation — real physics ([SUBSTRATE](docs/SUBSTRATE-validation.md)).**
TFIM ground states (n=16) compress **19× (critical) → 77× (paramagnet)** losslessly to
~1e-7, with four independent physics cross-checks (Page value, entropy monotonicity,
TT rank = Schmidt rank, honest decline). The Haar-random control does **not** compress
(**0.38×** — TT larger than dense); Blaze reports this instead of hiding it.

### Quickstart

```bash
pip install -e '.[quantum]'                              # numpy/scipy + cirq/quimb
pytest -q                                                # 33 gates

python -m blaze.examples.substrate_quantum_state         # flagship: 19–77× on TFIM
python -m blaze.examples.quantum_similarity_search       # Phase 7: search in compressed space
python -m blaze.examples.quantize_sweep                  # Phase 8: TT × quantization, with fidelity
```

```python
import numpy as np, blaze
psi = np.zeros(2**12, dtype=complex); psi[0] = psi[-1] = 2**-0.5   # a GHZ state
tt  = blaze.compress(psi.reshape((2,)*12), rel_tol=1e-10)          # TT/MPS
q   = blaze.quantize_tt(tt, bits=8)                                # second-stage codes
print(blaze.fidelity(tt, q.dequantize()))                         # ~1.0, no decompression
```

### Stack status

`Python` reference — all 8 phases · `Rust` `blaze-core` — Phases 2–6 (parity-gated) ·
`C++/CUDA` — Phase 3 SVD offload (cuSOLVER, f64+c64) · `Cirq` — Phase 5 bridge.
Phase 7–8 are the NumPy reference; their Rust port + CUDA batched overlap are the
documented next steps (ADR-0002).

