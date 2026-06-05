# Blaze — Phase 2 spec: Rust core (the implementation contract)

**Status:** Ready to build. Gate-checked against the Phase-1 Python reference.
**Date:** 2026-06-03

---

## 0. Locked scope (post-Phase-1 benchmark — READ FIRST)

The honest Phase-1 benchmark (fair baseline = best truncated SVD over **all** mode
bipartitions, at **matched parameter budget**) settled claim (b):

- **TT beats a fair matrix SVD only on genuinely TT-native data** — low rank across
  *every* cut, structure distributed over modes (e.g. a random low-bond MPS; a
  low-entanglement quantum state, guaranteed by physics).
- On generic "structured" data (smooth 4D fields, hyperspectral-like 4D, even a
  smooth high-order QTT signal), **a well-chosen single matrix SVD competes or wins**
  at equal params. "High-dimensional" or "smooth" is **not** enough.
- Random data: nobody compresses it (negative control passes).

**Therefore Blaze is NOT a general-purpose compressor, and must not claim to be.**
Its honest, usable value is:
1. **Quantum-state compression + verification** — Cirq circuit → statevector → MPS,
   with fidelity + MPS sampling. Physics guarantees TT-compressibility for
   low-entanglement states. This is the real differentiator.
2. **Honest compressibility diagnostics** — tell the user *whether* their data is
   TT-compressible (effective ranks / singular decay) **before** they trust it.
3. **Fast TT** for the cases where TT *is* the right tool.

Phase 2 builds a usable Rust core for exactly this. No overclaim, no CUDA yet.

## 1. Goal

A Rust crate **`blaze-core`** that mirrors the **validated Phase-1 Python API 1:1**
(the Python in `python/blaze/` is the executable golden reference — its *implementation*
is correct and tested; mirror it). CPU only (faer). Usable CLI + library + Python
bindings. Parity with Python is the primary gate.

## 2. What to build

### crate `blaze-core`
- **`TT`** struct: `cores: Vec<Array3<T>>` (use `ndarray`), `shape`, plus
  `ranks()`, `nparams()`, `reconstruct()`, `rel_error(orig)`, `compression_ratio(orig)`,
  `effective_ranks(tol)`, `singular_decay_summary(tols)` — same semantics as `tt.py`.
- **`tt_svd(tensor, max_rank, rel_tol)`**: successive matricization + truncated SVD via
  **`faer`** (pure-Rust linalg; real **and** complex), identical algorithm to `tt.py`
  (energy-budget truncation `eps = rel_tol / sqrt(ndim-1)` + `max_rank` cap; record full
  singular spectra for diagnostics).
- **dtypes:** `f64` and `c64` (complex is first-class — the quantum path needs it; faer
  supports it). `f32` optional after parity holds.
- **`compress(...) -> TT`** high-level entry, mirroring `blaze.compress`.

### CLI `blaze`
`blaze compress <in.npy> --max-rank R --rel-tol T -o out.blz` ·
`blaze reconstruct <out.blz> -o recon.npy` · `blaze info <out.blz>` ·
`blaze benchmark` (runs the **fair** classical benchmark — same verdicts as Python).

### `.blz` format
Header (magic `BLZ1`, dtype tag, ndim, shape, per-bond ranks) + cores row-major.
Document the byte layout in `docs/blz-format.md`. Python must be able to read it.

### Python bindings (maturin + PyO3, cdylib)
Expose the Rust core so `blaze.compress(..., backend="rust")` works and numpy arrays
pass with minimal copy. Python stays the orchestrator (Cirq, experiments).

## 3. Gates (the box checks — nothing advances until these pass)

1. **Parity / golden (the #1 gate):** for the same input + params, Rust `tt_svd`
   yields the **same ranks and the same rel-error as Python** (within fp tolerance).
   If it diverges, the Rust is wrong, not the Python.
2. **faer SVD validated** against a reference SVD on test matrices (real + complex).
3. **Honest benchmark parity:** `blaze benchmark` reproduces the Python fair-baseline
   verdicts (TT dominates only on TT-native data; ties on 4D structured; SVD competes
   on smooth QTT; random = nobody). Same conclusions, no regression to a strawman.
4. **Quantum round-trip:** a Cirq GHZ / low-entanglement statevector (c64) compresses
   to the **same fidelity** as Python (≈1.0 at the known low rank).
5. **Property test:** random TT → reconstruct → compress(tol) → `rel_error < tol`.
6. **CLI smoke:** compress→reconstruct round-trip on a `.npy` matches in-memory error.

## 4. Explicitly NOT in Phase 2
- **No CUDA** (Phase 3 — and only if the niche is worth accelerating; decide after Phase 2).
- No new algorithms (TT-cross / randomized / ALS) — TT-SVD only, parity with Python.
- No "general compressor" claims anywhere in code, docs, or CLI help.

## 5. Review discipline
This spec is gate-checked against the validated Python reference — especially **parity
with the Python** and **no scope creep back to a general compressor**. Honest
diagnostics + the quantum bridge are the product; speed is Phase 3.
