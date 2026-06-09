# Phase 8 — Second-stage core quantization (composable, honest)

**Date:** 2026-06-09 · Python reference path. Lossy-on-lossy, always reported against
the dense original — never a lossless codec (ADR-0002). Measured on the RTX 5060 Ti box.

## What Phase 8 is

A second, orthogonal compression stage on top of TT-SVD:

| Stage | Compresses | Mechanism |
|-------|------------|-----------|
| 1 (Phase 2) | **structure** | TT-rank truncation → ratio `R_tt` |
| 2 (Phase 8) | **residual numeric content** of the cores | each entry → b-bit code + scale |

Scheme: **symmetric affine** quantization, `code = round(x / scale)` clipped to
`±(2^(b−1)−1)`. Two granularities — `per_core` (one scale/core, minimal metadata) and
`per_bond` (one scale per right-bond column, better for the S·Vh core). Complex128
entries are quantized as separate real/imag codes. Code dtype is int8 (b≤8) or int16
(b≤16); **`nbytes()` reports the real bit-packed size** `ceil(n_codes·b/8) + scales`
(float32 scales). The errors of the two stages **compose**.

## Gate 1 — error is monotone in bits and composes correctly

Pure-quantization error (vs the unquantized TT) **strictly decreases** with bit width
(random complex core, per-bond). Combined error (vs the dense original) is always
`≥` the truncation-only floor — quantizing can only add — and **converges back to that
floor** as bits grow. Locked in `tests/test_phase8_quantize.py` (8 tests pass; full
suite **33 pass**).

A subtlety found and documented: a GHZ state's entries are exactly `±1/√2`, which
quantize *losslessly*, so its error sits at the **float32 scale floor** (~6e-8), not
the bit-width floor — a degenerate case. Monotonicity is asserted on spread-spectrum
states where the quantization step genuinely dominates.

## Gate 2 — golden: GHZ survives int8 at fidelity ~1

`fidelity(tt, quantize_tt(tt, bits=8).dequantize()) > 1 − 1e-3`, measured with the
**Phase 7 overlap** (in compressed space — the two phases compose).

## Gate 3 — the headline: TT-rank × quantization on real SUBSTRATE states

TFIM ground states, n=16, `rel_tol=1e-6` (the validated SUBSTRATE tolerance, so TT
ranks track **physical** entanglement). `R_tt` reproduces `docs/SUBSTRATE-validation.md`
exactly (19→77×). Fidelity is **measured**, not assumed:

| state | χ | R_tt | bits | R_total | ×extra | fidelity |
|-------|--:|-----:|-----:|--------:|-------:|---------:|
| TFIM h=1.0 (critical) | 14 | 19.3× | 8 | **131.9×** | 6.82× | 0.999794 |
| | | | 6 | 167.6× | 8.66× | 0.998369 |
| | | | 4 | 229.7× | 11.88× | 0.975722 |
| TFIM h=1.5 | 10 | 33.4× | 8 | **216.3×** | 6.48× | 0.999900 |
| | | | 4 | 363.6× | 10.90× | 0.983118 |
| TFIM h=2.0 | 7 | 60.2× | 8 | **364.1×** | 6.04× | 0.999856 |
| | | | 4 | 585.1× | 9.71× | 0.997293 |
| TFIM h=3.0 (paramagnet) | 6 | 76.6× | 8 | **447.3×** | 5.84× | 0.999943 |
| | | | 6 | 547.3× | 7.15× | 0.997936 |
| | | | 4 | 704.7× | 9.20× | 0.954422 |

**Read:** `R_total = R_tt × ×extra`. int8 multiplies the TT ratio by ~6× at fidelity
**0.9999** across the whole phase diagram; 4-bit reaches ~9–12× extra (up to **705×**
end-to-end) at a fidelity cost that is visible and reported (0.95–0.997). The deep
paramagnet at int8 is the sweet spot: **447× vs the dense statevector, fidelity 0.99994.**

## Scope and honest limits

- **`[KNOWN_LIMIT]` metadata erodes ×extra below the ideal.** The code bound is
  16 B → 2 B = 8× (int8) / 16× (4-bit). On these small-χ cores (χ=6–14) the per-bond
  scale metadata pulls int8 down to ~6×. `per_core` stores smaller on tiny cores;
  `per_bond` gives lower *error* on cores with a spread singular spectrum — a real
  trade-off, both schemes provided.
- **float32 scale floor (~6e-8).** Plenty for b≤~12; beyond that the scale precision,
  not the bit width, would cap accuracy. Documented, not hidden.
- **4-bit is genuinely lossy.** Fidelity drops to ~0.95 on the paramagnet — fine for
  similarity/triage, not for high-precision reconstruction. The user picks the point.
- **Reference path only.** NumPy reference; a `.blz` v2 with physically packed nibbles
  and the Rust port are deferred (ADR-0002).

## Reproduce

```bash
pip install -e '.[quantum]'                 # quimb for the TFIM path
python -m blaze.examples.quantize_sweep
pytest tests/test_phase8_quantize.py -v     # 8 gates
```

## Next

- **`.blz` v2:** persist quantized TTs with physically bit-packed codes (the format
  already isolates cores; add a `dtype_tag` for quantized payloads).
- **Approximate overlap (Phase 7 × 8):** run the `TTIndex` zipper directly on
  quantized cores → cheaper, approximate similarity search, with the error budget
  measured here. This is the full vector-search analogue: search over quantized codes.
- **Rust port:** `quantize`/`dequantize` parity in `blaze-core`, gated vs this reference.
