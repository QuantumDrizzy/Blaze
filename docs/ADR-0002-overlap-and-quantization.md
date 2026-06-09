# ADR-0002: Overlap-in-compressed-space + core quantization (Phases 7–8)

**Status:** Accepted (Phase 7 implemented; Phase 8 specified)
**Date:** 2026-06-09
**Deciders:** Antonio (QuantumDrizzy)
**Extends:** ADR-0001 (architecture + honesty contract — still binding)

---

## Context

Blaze (ADR-0001) compresses high-order data to TT/MPS and reconstructs it. Phases
1–6 closed: TT-SVD, CUDA SVD, approximate reconstruction, MPS↔circuit synthesis,
Rust CLI + `.blz`. Validated on real SUBSTRATE quantum states (19–77× on physical
TFIM ground states; honest decline on Haar volume-law control).

A study of vector-search libraries (the TurboQuant/TurboVec line: random rotation +
per-coordinate Lloyd–Max quantization + inner-product estimation **directly on the
compressed codes**, never decompressing) exposed two capabilities Blaze did *not*
have, both of which are natural for a tensor-network compressor:

1. **Similarity / inner products computed in the compressed representation**, without
   ever materializing the dense object. Blaze already had the math latent — an MPS
   overlap is a cheap network contraction — but the public API only offered
   `reconstruct()` + dense `vdot` (O(2ⁿ)). Even `blaze.cirq.fidelity` densified.
2. **A second, orthogonal compression stage: quantizing the stored core entries**
   (the TT-rank stage compresses *structure*; quantization compresses the *residual
   numeric content* of the cores).

This ADR adopts both as Phases 7 and 8. We do **not** vendor TurboVec, depend on it,
or clone its algorithm. We take the *idea that compressed-space operation is the
differentiator* and realize it the tensor-network-native way.

## Decision

### Phase 7 — Inner product, norm, fidelity, distance, and search on the TT directly

The primitive is the standard MPS **zipper** contraction of ⟨A|B⟩ through a transfer
matrix `E` of shape `(r_A, r_B)`:

```
E ← [[1]]                                  # (1,1)
for A_k, B_k in zip(a.cores, b.cores):     # A is the bra (conjugated)
    T ← tensordot(E,        B_k, [[1],[0]])      # (aL, d, bR)
    E ← tensordot(conj(A_k), T,  [[0,1],[0,1]])  # (aR, bR)
⟨A|B⟩ ← E[0,0]
```

Cost **O(n · d · χ³)** vs **O(dⁿ)** for the dense inner product. Derived, all without
decompression:

| API | Definition |
|-----|------------|
| `inner(a, b)` | ⟨a\|b⟩ (bra conjugated) |
| `norm(a)` | √Re⟨a\|a⟩ |
| `fidelity(a, b)` | \|⟨a\|b⟩\|² / (⟨a\|a⟩⟨b\|b⟩) ∈ [0,1], normalization-free |
| `distance(a, b)` | ‖a−b‖ = √(⟨a\|a⟩ + ⟨b\|b⟩ − 2 Re⟨a\|b⟩) |
| `TTIndex` | a database of TTs; top-k search by fidelity/distance, **entirely in compressed space** |

`TTIndex` is the differential: **exact** similarity search over compressed states
without decompression — stronger than FAISS-style PQ (which is lossy/approximate),
*when the data is genuinely TT-structured*. That qualifier is the honest boundary,
identical to ADR-0001's: it is cheap and exact only where χ stays small.

### Phase 8 — Core quantization (second compression stage)

After TT-SVD truncates rank (compressing structure), quantize the residual core
entries to low-bit codes (int8 → 4-bit), per-core calibrated, with a documented
reconstruction-error budget that *composes* with the truncation error. This is the
TurboVec idea (calibrated scalar quantization of the stored numbers) applied to TT
cores instead of raw vectors. Specified here; implemented after Phase 7 is green.

## Options Considered

### Phase 7 placement
- **Python reference first, then Rust port, then (maybe) CUDA (chosen)** — matches the
  ADR-0001 build discipline that delivered Phases 2–6. Reference + golden gates lock
  correctness; the Rust/CUDA port is a parity exercise, not a rewrite.
- **Straight to CUDA** — rejected. ADR-0001 §Action Items: no kernels before the
  reference proves and a benchmark justifies them. A single O(nχ³) overlap on a
  low-χ 1D state is CPU-bound (host↔device transfer dominates), exactly as the
  SUBSTRATE-validation note found for skinny SVDs. The GPU pays for **batched** search
  over a large `TTIndex` (many overlaps at once) — deferred until that workload exists.

### Search semantics
- **Exact overlap contraction (chosen)** — no approximation; the only error is fp.
  The honest, differentiating claim: exact NN search in compressed space for
  TT-structured data.
- **Approximate (sketch/quantized overlap)** — deferred to Phase 8 (it composes:
  quantized cores → approximate but cheaper overlaps, with a measured error budget).

## Honesty gates (must pass — non-negotiable, per ADR-0001)

Phase 7:
1. `inner(a,b)` equals the dense `vdot(reconstruct(a), reconstruct(b))` to ≤1e-12
   (random shallow states, several n). The zipper computes the *same number* the
   dense path would — without densifying.
2. `fidelity`/`distance` match their dense definitions to ≤1e-12.
3. Golden anchors: ⟨GHZ|GHZ⟩=1, fidelity(GHZ, |0…0⟩)=½, ⟨0…0|1…1⟩=0.
4. **Scaling proof:** runs at n≥40 for low-χ states (2⁴⁰ dense is impossible),
   confirming it never decompresses.
5. `[KNOWN_LIMIT]` documented: cost is O(nχ³); for volume-law states χ=d^{n/2} and the
   overlap is as expensive as dense. No claim survives outside the low-χ regime.

Phase 8: quantization error is measured, composes with truncation error, and is
reported against the same dense baseline — never hidden, never vs a lossless codec.

## Consequences

**Easier:** Blaze becomes a *queryable* compressed store, not just compress/restore —
"a similarity index for tensor-network-structured data" is a real, differentiated,
job-relevant artifact. Kills the dense path in `cirq.fidelity`.
**Harder:** must keep resisting the overclaim — this is exact only for low χ; it is
**not** a general vector DB and must never be sold as a FAISS replacement for generic
embeddings (those are not TT-structured). The `TTIndex` docstring carries that limit.
**Revisit when:** a real batched-search workload appears → benchmark CUDA batched
overlap; if Phase 8 quantization doesn't add ratio at acceptable error on real
SUBSTRATE states, scope it down and say so.

## Action Items
1. [x] `python/blaze/overlap.py`: `inner/norm/fidelity/distance/TTIndex` (Phase 7).
2. [x] Golden + scaling gates in `tests/test_phase7_overlap.py`.
3. [x] `examples/quantum_similarity_search.py` — search over SUBSTRATE TFIM states.
4. [x] `docs/PHASE7-results.md` with measured numbers.
5. [ ] Rust port (`blaze-core`): `inner/fidelity` parity with Python (Phase 7b).
6. [ ] CUDA batched overlap for `TTIndex` — only once a large-DB workload justifies it.
7. [ ] Phase 8: calibrated core quantization (int8→4-bit), composed error budget.
