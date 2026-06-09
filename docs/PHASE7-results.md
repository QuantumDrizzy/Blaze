# Phase 7 — Inner product, fidelity & similarity search in compressed space

**Date:** 2026-06-09 · Python reference path. **No advantage claim** beyond the
measured exactness + scaling (ADR-0002). Hardware: RTX 5060 Ti box (CPU path here).

## What Phase 7 is

Operations on the TT/MPS **without decompressing it**. The primitive is the MPS
"zipper": the overlap ⟨A|B⟩ is contracted site by site through a transfer matrix
`E` of shape `(r_A, r_B)`, cost **O(n·d·χ³)** instead of **O(dⁿ)**:

```
E ← [[1]]                                       # (1,1)
for A_k, B_k in zip(a.cores, b.cores):          # A is the bra (conjugated)
    T ← tensordot(E,        B_k, [[1],[0]])     # (aL, d, bR)
    E ← tensordot(conj(A_k), T,  [[0,1],[0,1]]) # (aR, bR)
⟨A|B⟩ ← E[0,0]
```

From it: `inner`, `norm`, `fidelity`, `distance`, and `TTIndex` (top-k similarity
search). This replaces the dense `O(2ⁿ)` path that `blaze.cirq.fidelity` used
(`reconstruct()` then `vdot`).

## Gate 1 — the zipper computes the SAME number as the dense product (exact)

Measured: `inner(a,b)` vs `vdot(reconstruct(a), reconstruct(b))`, GHZ × random,
complex128. The agreement is at floating-point machine precision, and the zipper
pulls ahead as the dense `2ⁿ` work grows.

| n | χ | zipper (ms) | dense (ms) | \|zipper − dense\| |
|--:|--:|------------:|-----------:|-------------------:|
|  8 |  16 | 0.452 | 0.279 | 4.5e-16 |
| 10 |  32 | 0.352 | 0.343 | 6.0e-16 |
| 12 |  64 | 0.522 | 2.230 | 1.2e-15 |
| 14 | 128 | 1.112 | 8.444 | 1.1e-15 |

The random ket here is **volume-law** (χ hits the 2^(n/2) ceiling), the *worst*
case for the zipper — yet it already wins by n=12 because the dense path pays the
full `2ⁿ` reconstruction. For genuinely low-χ data the gap is unbounded (Gate 3).

## Gate 2 — golden anchors (exact)

`⟨GHZ|GHZ⟩ = 1`; `fidelity(GHZ, |0…0⟩) = ½` (since `⟨GHZ|0…0⟩ = 1/√2`);
`⟨0…0|1…1⟩ = 0`; `fidelity` and `distance` match their dense definitions to ≤1e-12.
Locked in `tests/test_phase7_overlap.py` (14 tests, all pass; full suite 25 pass).

## Gate 3 — scaling proof: it never decompresses

n = **40** qubits, product states built directly as bond-1 TTs. A dense complex128
statevector would be `2⁴⁰ × 16 B ≈ 17.6 TB` — impossible to even allocate.

```
fidelity(a, a) = 1.000000000000   inner(a, c) = 0.00e+00   norm(a) = 1.000000000000
                                                       in 3.6 ms
```

`c` differs from `a` by one orthogonal single-qubit rotation on the last site, so
`⟨a|c⟩ = 0` exactly — recovered instantly, with no `2⁴⁰` array ever formed.

## Gate 4 — the differential: similarity search in compressed space (real physics)

`examples/quantum_similarity_search.py` builds a database of **TFIM ground states**
across the phase diagram (fields `h = 0.6 … 3.0`, built with quimb — SUBSTRATE's
own stack), compresses each (`n=16`, ~12× smaller, one TT ≈ 82 KiB vs 1 MiB dense),
indexes them in a `TTIndex`, and queries with a **held-out** state at `h = 1.95`.

Retrieval (query took **5.95 ms**, all in compressed space, nothing decompressed):

```
1.  h=1.8   fidelity=0.997702      <- correct nearest neighbour
2.  h=2.2   fidelity=0.995978
3.  h=2.6   fidelity=0.981345
```

Pairwise fidelity matrix (every entry an exact overlap on the TTs directly):

```
         0.60   1.00   1.40   1.80   2.20   2.60   3.00
 0.60   1.000  0.414  0.114  0.060  0.041  0.031  0.026
 1.00   0.414  1.000  0.805  0.654  0.565  0.506  0.466
 1.40   0.114  0.805  1.000  0.964  0.912  0.868  0.833
 1.80   0.060  0.654  0.964  1.000  0.988  0.966  0.945
 2.20   0.041  0.565  0.912  0.988  1.000  0.995  0.984
 2.60   0.031  0.506  0.868  0.966  0.995  1.000  0.997
 3.00   0.026  0.466  0.833  0.945  0.984  0.997  1.000
```

**This is not just a search demo — it is correct physics.** The fidelity collapses
between the ordered side (`h=0.6`) and the paramagnet (`h≥1.8`): the `h=0.6` ground
state is nearly orthogonal (fidelity 0.03–0.06) to the high-field states, while the
paramagnetic states are mutually similar (fidelity >0.94). That sharp drop is the
**ground-state-fidelity signature of the quantum phase transition** (Zanardi &
Paunković, PRE 2006) — recovered here purely from overlaps computed in compressed
space, never touching a dense statevector.

## Scope and honest limits

- **`[KNOWN_LIMIT]` cost is O(n·χ³).** It is cheap and exact *only* for low χ
  (compressible / low-entanglement data). A volume-law (Haar) database has
  χ ~ d^(n/2) and the overlap costs as much as the dense product — Gate 1's n=8/10
  rows already show the zipper losing its edge as χ saturates. `TTIndex` is **not**
  a general-purpose vector DB: generic high-entropy embeddings are not TT-structured.
- **Exact, not approximate.** Unlike PQ-style vector quantization, there is no
  lossy codebook here; the only error is floating point. (Approximate-but-cheaper
  overlaps via quantized cores are Phase 8.)
- **Reference path only.** This is the NumPy reference. Rust parity (`blaze-core`)
  and CUDA **batched** overlap (where the GPU pays — many overlaps per query over a
  large index) are deferred follow-ups, not yet built (ADR-0002 items 5–6).

## Reproduce

```bash
pip install -e '.[quantum]'                       # quimb for the TFIM path
python -m blaze.examples.quantum_similarity_search
pytest tests/test_phase7_overlap.py -v            # 14 gates
```

## Next

- **Phase 7b:** port `inner`/`fidelity` to `blaze-core` (Rust), parity-gated vs this
  reference (same discipline as Phases 2–6).
- **CUDA batched overlap:** one query vs an N-state index is N independent zippers —
  the batched workload where the GPU finally beats the CPU. Benchmark when a large
  index exists.
- **Phase 8:** calibrated core quantization (int8 → 4-bit) as a second compression
  stage, with an error budget that *composes* with the TT truncation error.
