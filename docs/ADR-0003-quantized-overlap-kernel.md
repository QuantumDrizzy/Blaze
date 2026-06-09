# ADR-0003: Quantized Overlap Kernel (Phase 9-real)

**Status:** Proposed
**Date:** 2026-06-09
**Deciders:** Antonio

## Context

Phases 7b (Rust zipper `inner`/`fidelity`/`distance`, parity 3e-15) and 8b (Rust
`quantize_tt`/`QuantizedTT`, codes bit-exact vs Python) are done and parity-gated.
Phase 9-Python validated that similarity search over **quantized** cores holds:
recall@k = 1.0 @8-bit, ~0.90 @4-bit, at ~7x memory over the float TT.

What is NOT done: the **compute/throughput win** — an int8 kernel. This ADR decides
*how* to get it without building something that doesn't pay.

## The problem: the zipper is NOT a flat int8 GEMM

Per site the zipper does two contractions:

```
T[aL,d,bR] = Σ_bL  e[aL,bL] · B[bL,d,bR]            # e is FLOAT, B is int8
e'[aR,bR]  = Σ_aL,d  conj(A[aL,d,aR]) · T[aL,d,bR]
```

The trap: the **transfer matrix `e` is inherently float** — it accumulates products
of heterogeneous scales along the chain. So each contraction is **float × int8**, not
int8 × int8. VNNI (`vpdpbusd`) and int8 tensor cores need **int8 × int8 → int32**;
they do not apply directly.

To get pure int8×int8 you must **factor the scales out** and accumulate codes in
integers. That only works with **per_core** quantization (one scale per core); with
**per_bond** the per-column scales interleave along the chain and do not factor. And
per_core gives **worse recall** (measured in Phase 8). On top of that, accumulating
int8 codes across n sites with bond χ **overflows int32** quickly → needs i64 + care.

**Conclusion:** the *safe* int8 win in the zipper is **memory/bandwidth** (read int8,
~8x less core traffic), not integer compute. The VNNI/tensor-core win is possible but
restricted (per_core, overflow) and **may not pay if the overlap is already
memory-bound** — which at small χ it is.

## Options

### Route A — dequant-per-core + float zipper (SIMD)
Read int8 codes, expand per-core to f32/f64 with SIMD (pulp), run the existing zipper.
- **Gain:** memory/BW (up to ~8x less core traffic streamed from RAM/L2).
- **Cost:** low — composes 7b + 8b.
- **Constraint:** none; works with per_bond. Compute stays float.

### Route B — integer accumulation + VNNI (per_core only)
Factor scales out, accumulate codes in i32/i64, `vpdpbusd` for the int8 dot products,
apply the scale product at the end.
- **Gain:** integer compute (≈4x int8 ops/cycle vs FMA).
- **Cost:** high.
- **Constraint:** per_core only · overflow → i64 + normalization · worse recall.

### Route C — CUDA batched N-overlap
The real TTIndex use case: N indexed states, 1 query → N overlaps in parallel
(one block/warp per overlap). int8 cores are cache-resident (fits the 32 MB L2 →
dodges the ~920-cycle DRAM wall, the CYBERDECK thesis).
- **Gain:** scale (large N) + cache residency.
- **Cost:** very high (kernel + nvcc build + parity).
- **Constraint:** the large batch IS the use case; not worth it for single overlaps.

| Route | Gain | Cost | Catch |
|-------|------|------|-------|
| **A** dequant + float zipper (SIMD) | memory/BW (~8x less traffic) | low | none; per_bond OK |
| **B** integer accum + VNNI | compute (~4x int8/cycle) | high | per_core only · i64 overflow · worse recall |
| **C** CUDA batched | scale (large N) + L2-resident | very high | batch is the use case |

## Decision

**Route A first, then a roofline measurement that gates B/C.**

Building a VNNI or CUDA int8-compute kernel *before* knowing whether the zipper is
compute- or memory-bound would be inflated work. Route A is also the baseline any
later kernel must beat, so it is required regardless.

## The deciding metric

**Is the zipper memory-bound or compute-bound at the χ that matter?** → a roofline of
the overlap: arithmetic intensity AI(χ) = FLOPs / bytes streamed.

- **Memory-bound** (likely at χ ≤ ~64): Route A captures the win; the VNNI/CUDA
  int8-compute kernel does **not** pay → stop and document.
- **Compute-bound** (large χ, many states): B or C are justified.

## Implementation plan — Route A (pick up here)

1. **`crates/blaze-core/src/quantized_search.rs`** (new):
   - `inner_quantized_c64(qa: &QuantizedTT, qb: &QuantizedTT) -> Complex64`:
     dequant per-core into **reused scratch buffers** (avoid per-call allocation),
     run the zipper from `lib.rs`. SIMD int8→f32 via `mapv` / pulp.
   - `QuantizedTTIndex { items: Vec<QuantizedTT>, self2: Vec<f64>, labels, metric }`
     with `add` / `add_tt` / `query` mirroring the Python `QuantizedTTIndex`.
   - Wire `mod quantized_search; pub use …` in `lib.rs`.

2. **Micro-benchmark** (`#[ignore]` timing test or criterion):
   zipper on **f64 dense cores** vs **dequant-int8** zipper, across
   χ ∈ {8, 16, 32, 64, 128}. Measure ns/overlap + bytes streamed → AI + effective
   GB/s. Print the crossover. This is the number that decides B/C.

3. **Parity test:** `inner_quantized_c64` score == Python `QuantizedTTIndex` score on
   the same data (via `blaze-py` binding, or a hardcoded anchor like Phase 8b).

4. **Decision point:** if memory-bound at χ ≤ 64 → **stop**, Route A is the win,
   document it in `docs/PHASE9-results.md`. If compute-bound → proceed to B (VNNI
   per_core, i64 accum) or C (CUDA batched, CYBERDECK).

## Consequences

- A is low-risk, composes existing parity-gated code, gives a measurable memory-BW
  win **and** the baseline.
- The honest outcome may be "the int8-compute kernel does not pay" — a valid,
  documented result, not a failure.
- B restricts to per_core (worse recall); C is the real scale play (TTIndex batched
  on the GPU — overlaps with CYBERDECK's batched-overlap idea).

## Honesty contract

Every speedup stated as (path, baseline): e.g. "int8 dequant zipper vs f64 zipper,
CPU, χ=N, kernel-only". No int8-**compute** number is claimed until the VNNI/CUDA path
exists and is parity-gated against the f64 zipper.

## Status of the phase ladder (as of this ADR)

```
✅ 7b  zipper inner/fidelity/distance — Rust (f64+c64), parity 3e-15
✅ 8b  quantize_tt + QuantizedTT — Rust, codes bit-exact vs Python
⬜ 9A  dequant-per-core + float zipper (Rust, SIMD) + roofline   ← NEXT
⬜ 9B  integer accumulation + VNNI (per_core)        [gated on roofline]
⬜ 9C  CUDA batched N-overlap (int8, L2-resident)    [gated on roofline]
```
