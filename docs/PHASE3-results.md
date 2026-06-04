# Phase 3 — GPU SVD offload (cuSOLVER): results

**Date:** 2026-06-04 · RTX 5060 Ti (sm_120), CUDA 13.0, cuSOLVER.

## What was built
- `cuda/blaze_svd.cu` — C-ABI economy SVD via **cuSOLVER** (`gesvd`), handling the
  row-major (ndarray) ↔ column-major (cuSOLVER) conversion and the wide-matrix (m<n)
  case. **Layout verified standalone**: tall 4×2 and wide 2×4 reconstruct to ~1e-16.
- `crates/blaze-core/build.rs` — compiles the `.cu` with `nvcc -arch=sm_120` into a
  static lib and links cuSOLVER + cudart, **only with `--features cuda`** (CPU path
  via `nalgebra` stays the default).
- `compress_f64_cuda` — the SVD step of TT-SVD offloaded to the GPU; everything else
  (matricization, truncation, core assembly) stays identical to the CPU reference.

## Gate 1 — parity (verified)
`cargo test -p blaze-core --features cuda cuda_parity -- --nocapture`

CPU vs GPU on the same tensor → **identical ranks**, matching `rel_error`
(CPU 2.81e-15 / GPU 3.48e-16, both exact for that input). The GPU path produces the
same decomposition as the validated CPU core.

## Gate 2 — timing (HONEST)

⚠️ **A debug build is misleading here.** In debug, `nalgebra` (CPU) is unoptimized and
~50× slower than release, which fakes a "56× speedup". The honest comparison is
**release CPU vs GPU**:

| 4D tensor (n⁴) | elements | CPU release (ms) | GPU (ms) | speedup | winner |
|----------------|----------|------------------|----------|---------|--------|
| n=8  | 4 096     | 0.7   | 27.0  | 0.03× | **CPU** (transfer/launch-bound) |
| n=16 | 65 536    | 25.6  | 30.6  | 0.84× | CPU |
| n=24 | 331 776   | 106.3 | 81.3  | 1.31× | **GPU** ← crossover |
| n=32 | 1 048 576 | 354.0 | 122.0 | **2.90×** | **GPU** |

**Honest read:** GPU SVD offload gives a **modest ~3× at ~1M elements**, with the
**crossover near n≈24 (~330k)**. Below it the **CPU wins** — host↔device transfer and
kernel-launch overhead dominate small problems. `[KNOWN_LIMIT]`.

**Why it's conservative:** the MVP copies each matricization host→device→host per SVD.
A GPU-resident pipeline (keep tensors on the device across the successive SVDs) would
cut that transfer and widen the win — that is the next optimization, not done here.

## c64 — the quantum path (added 2026-06-04)
`cusolverDnZgesvd` + the conjugate-transpose handling for the wide case, verified
**standalone first** (complex tall 3×2 + wide 2×3 reconstruct to ~1e-16). Then
`compress_c64_cuda` parity vs the CPU reference on a *non-trivial* truncation:
**identical** — CPU `rel_err = 7.094302e-4` == GPU `7.094302e-4`, same ranks
`[1,3,3,1]`. This is the highest-value GPU piece: large quantum statevectors → MPS,
where the SVDs are genuinely large and the GPU pays off most.

**Timing (release, honest)** — c64 wins by *more* than f64 (complex SVD is ~4× the
flops, so the GPU's compute advantage dominates the transfer overhead more):

| 4D complex (n⁴) | elements | CPU release (ms) | GPU (ms) | speedup | winner |
|-----------------|----------|------------------|----------|---------|--------|
| n=8  | 4 096     | 1.6   | 20.4  | 0.08× | CPU (transfer-bound) |
| n=16 | 65 536    | 32.8  | 69.5  | 0.47× | CPU |
| n=24 | 331 776   | 229.6 | 76.4  | 3.01× | **GPU** ← crossover |
| n=32 | 1 048 576 | 844.7 | 221.7 | **3.81×** | **GPU** (= a 20-qubit statevector) |

The quantum path gets **~3.8× at a 20-qubit statevector** — more than the f64 2.9×, as
expected. Same honest crossover (~n≈24); below it the GPU loses (`[KNOWN_LIMIT]`).

## Scope (unchanged, honest)
This accelerates the **validated niche** — large TT-native / quantum-state tensors,
where TT is the right tool *and* the matrices are big enough for the GPU to pay off.
It does **not** make Blaze a fast general compressor (Phase-1 benchmark settled that).
The biggest real upside is the quantum path (large statevectors → MPS), where the
SVDs are genuinely large — c64 GPU SVD is the natural next step.

## Discipline note
The 56× (debug) → 2.9× (release) gap is the "distrust a too-good benchmark" rule
applied to our own result: always state the baseline (release, `nalgebra` CPU) and
the regime (large tensors), never a bare speedup.
