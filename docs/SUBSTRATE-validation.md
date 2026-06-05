# SUBSTRATE validation — Blaze on real quantum many-body states

**Date:** 2026-06-04 · quimb (SUBSTRATE Phase-A TN stack) → Blaze TT-SVD (Python reference path).

## What this validates

The Blaze thesis for the ecosystem: it is a **specialist** compressor for the
structured / low-entanglement data the projects produce, with an honest
diagnostic that says when *not* to use it. This is the first test on **real
physics data** rather than synthetic tensors.

SUBSTRATE simulates quantum many-body spin systems; its Phase-A tensor-network
stack is quimb (`cryptotn_gpu/cryptotn`, `quantum_lab/P3_G2/tn_quimb.py`). We
build dense ground states of the transverse-field Ising model (TFIM)

```
H = -J Σ Z_i Z_{i+1}  -  h Σ X_i
```

with that same stack, reshape the `2^n` statevector into a `(2,)^n` tensor, and
compress with Blaze. Because of the reshape, **Blaze's TT bond dimensions ARE
the MPS / Schmidt ranks of the wavefunction** — the compression is literally the
matrix-product-state factorization of the physical state.

## Result — n = 16 spins (2¹⁶ = 65,536 complex amplitudes, rel_tol = 1e-6)

| state | S (bits) | Schmidt rank | TT max-rank | params | ratio | rel. error | verdict |
|-------|---------:|-------------:|------------:|-------:|------:|-----------:|---------|
| TFIM h=1.0 (critical)   | 0.611 | 13 | 14 | 3 388 | 19.3× | 4.3e-7 | compress |
| TFIM h=1.5              | 0.221 |  9 | 10 | 1 964 | 33.4× | 4.3e-7 | compress |
| TFIM h=2.0              | 0.128 |  6 |  7 | 1 088 | 60.2× | 4.0e-7 | compress |
| TFIM h=3.0 (paramagnet) | 0.062 |  5 |  6 |   856 | 76.6× | 9.3e-8 | compress |
| **Haar-random (control)** | 7.279 | 256 | 256 | 174 760 | **0.38×** | 7e-15 | **decline (TT > dense)** |

The original statevector is 65,536 × complex128 = 1.0 MiB. The physical states
compress to 1–5 % of that, losslessly to ~1e-7. The Haar control does **not**
compress — its TT needs 2.7× the dense storage.

## Why these numbers are trustworthy (not inflated)

Four independent physics cross-checks, applied in the spirit of *distrust a
too-good result*:

1. **Page value.** The Haar control's half-cut entropy is 7.279 bits. Page's
   formula for `d_A = d_B = 256` gives `⟨S⟩ = ln(256) − 1/2 = 7.28 bits`. The
   match confirms the control is genuinely volume-law and the entropy
   computation is correct.
2. **Monotonicity.** Entanglement `S` falls 0.611 → 0.062 from criticality into
   the paramagnet; compression rises 19× → 77× in lockstep. No anomalies.
3. **TT rank = Schmidt rank (+1).** The `+1` is the tolerance distribution
   (`eps = rel_tol/√(n−1) ≈ 2.6e-7` per bond is stricter than the 1e-6 cut), so
   one extra singular value is kept. Explained, not magic — the compression *is*
   the MPS factorization.
4. **Honest decline.** The Haar state's TT is 2.7× *larger* than the dense array
   (ratio 0.38×). Blaze reports this instead of hiding it, and the
   `analyze_compressibility` diagnostic flags "high effective ranks" *before* the
   full decomposition is paid for.

## Scope and honest limits

- **n = 16 is small.** Even the critical state is only rank ~14 here; at larger
  `n` the critical bond dimension grows (log-corrected area law) and the ratio
  drops, while the paramagnet stays cheap. The *relative* story (ratio tracks
  `S`) is size-independent; the *absolute* ratios are `n`-dependent. `[KNOWN_LIMIT]`
- **CPU vs GPU.** This used the Python reference path (correctness + ratio are
  device-independent). The c64 **GPU** SVD (`PHASE3-results.md`, ~3.8×) pays off
  only when the SVD matrices are large, i.e. at **large bond dimension**
  (χ ~ hundreds–thousands: 2D systems, or `cryptotn_gpu`'s χ = 2500 TDVP target).
  For these small-χ 1D ground states the SVDs are skinny and the CPU already
  wins — do **not** expect the 3.8× here. `[KNOWN_LIMIT]`
- This is **not** evidence that Blaze is a general compressor (PHASE1 settled
  that). It is evidence that Blaze is the right tool for SUBSTRATE-class data.

## Reproduce

```bash
pip install -e '.[quantum]'      # numpy, scipy, quimb
python -m blaze.examples.substrate_quantum_state
```

Statevectors are written to `data/substrate/*.npy` (gitignored). The honest
invariants — paramagnet compresses, Haar declines, ratio tracks entropy, TT rank
≈ Schmidt rank — are locked as a regression test in
`tests/test_substrate_validation.py` (skips cleanly if quimb is absent).

## Next (deferred)

- **GPU-relevant regime:** a large-χ state (2D lattice / time-evolved) run
  through the Rust/CUDA c64 path, where the GPU SVD actually pays off.
- **Further ecosystem adapters:** HELIOS telemetry, neural-network activations.
