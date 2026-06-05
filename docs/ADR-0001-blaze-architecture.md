# ADR-0001: Blaze — Tensor-Network compression of high-order data

**Status:** Proposed (architecture spec — the contract the implementation must satisfy)
**Date:** 2026-06-03
**Deciders:** Antonio (QuantumDrizzy)

---

## Context

Blaze is a **lossy compressor + approximate reconstructor for high-order numerical
data**, using **Tensor-Train / Matrix-Product-State (TT/MPS) decomposition** as the
compression mechanism, accelerated on the GPU, with a **load-bearing quantum layer**
via Cirq. Target hardware: RTX 5060 Ti 16GB (sm_120) + Xeon 16c/32t.

This ADR sets the **architecture and — critically — the honesty contract** that every
phase of the implementation must satisfy. Nothing ships that fails the gates below.

### The honest framing (read this first — it is the whole point)
TT/MPS is real, proven math (Oseledets 2011; quantum many-body physics). But it is
**not a general-purpose compressor**, and pretending otherwise is the trap:

- **TT/MPS compresses well *only* when the data has low TT-rank structure** — i.e.
  correlations / low-dimensional structure across modes. Scientific tensors, feature/
  embedding tensors, simulation fields, multi-way arrays with redundancy: yes.
  Generic high-entropy bytes: **no** — the TT-rank explodes and there is *no*
  compression. This `[KNOWN_LIMIT]` is documented, not hidden.
- **Blaze is LOSSY** (ranks are truncated → reconstruction error). The honest
  comparison is therefore **vs other lossy high-order methods** — truncated SVD/PCA,
  Tucker, autoencoders — **NOT vs Zstd/Parquet** (lossless; apples-to-oranges).
- The genuine niche where TT *wins*: **order-≥3 tensors** where a single reshaped
  SVD/PCA can't exploit cross-mode structure. That is where Blaze must prove itself.
- **The "quantum" is already in the method:** an MPS *is* a quantum many-body state.
  That is the honest quantum root — not a bolted-on circuit. Cirq's job (below) is to
  make that correspondence **explicit and verifiable**, not decorative.

## Decision

Build Blaze as a **standalone project** (its own folder/repo — separate from
SUBSTRATE/TESSERA by design, though their proven MPS-CUDA work is a *reference* to
avoid reinventing). Stack, each language where it fits:

| Layer | Language | Role |
|-------|----------|------|
| Ingestion + tensorization, CLI, persistence, orchestration | **Rust** | load data, reshape to high-order tensors, the usable `blaze` CLI |
| TT/MPS decomposition + contraction kernels (the hot core) | **C++/CUDA** | TT-SVD, rank-truncated contraction, GPU throughput |
| TT algorithms, experiments, benchmarks, the quantum layer | **Python** | reference impl, ratio/error/throughput benchmarks |
| MPS ↔ quantum-circuit bridge | **Cirq** | construct + verify the state-prep circuit for a compressed MPS |

### The Cirq decision (load-bearing, honest)
A low-bond-dimension MPS can be prepared by a shallow quantum circuit. **Cirq's role:
given a TT/MPS produced by the compressor, construct the corresponding
state-preparation circuit and verify — by simulation, on small instances — that the
circuit reproduces the MPS amplitudes.** This makes the chain
*data → tensor network → quantum circuit* concrete and checkable. It is:
- **Load-bearing** — a real, verifiable artifact tying the quantum layer to the
  compression method (not feature-mapping decoration).
- **Honest** — small-scale, simulated, with **no speedup or advantage claim**. The
  deliverable is the *correspondence + verification*, plus an honest research question
  (does circuit depth track compressibility?).

## Options Considered

### Compression mechanism
- **TT/MPS (chosen)** — exploits cross-mode low-rank structure of high-order tensors;
  reuses the ecosystem's tensor-network muscle. Lossy; niche but real.
- **Truncated SVD/PCA / Tucker** — the honest *baselines*, not the product. Blaze must
  beat these on order-≥3 structured tensors to justify itself.
- **Classical codecs (Zstd/Parquet)** — different problem (lossless, byte-level).
  Used only as a sanity reference, never as the headline comparison.

### Cirq role
- **MPS↔circuit bridge (chosen)** — honest, load-bearing, tied to the method.
- **Quantum feature mapping** — rejected: no clear link to *compression*; decoration risk.
- **Quantum optimization of TN structure** — deferred: speculative, simulator-bound,
  high decoration risk. Revisit only if a concrete sub-problem demands it.

## Build sequence (honesty-gated)

| Phase | Build | Stack | Gate (must pass before next) |
|-------|-------|-------|------------------------------|
| 1 | Ingestion + tensorization | Rust + Python | loads ≥1 real high-order dataset; reshape is lossless + reversible |
| 2 | TT/MPS compression (TT-SVD + rank truncation) | Python + CUDA | **benchmark ratio @ reconstruction error vs truncated-SVD/PCA on STRUCTURED data; document the generic-data `[KNOWN_LIMIT]`** |
| 3 | CUDA kernels (TT contraction/decomposition) | Rust/C++ + CUDA | GB/s throughput; bit-for-bit validated against the Python reference |
| 4 | Approximate reconstruction | Python + CUDA | reports relative Frobenius error; error↔rank curve is monotone + honest |
| 5 | Cirq MPS↔circuit bridge | Cirq + Python | circuit reproduces the MPS amplitudes on small n (verified by simulation) |
| 6 | Rust CLI + persistence | Rust | `blaze compress/decompress <file>` works end-to-end on the target HW |

## Consequences

**Easier:** a genuinely differentiated, job-relevant artifact (TT + CUDA + honest
quantum) that reuses real expertise.
**Harder:** must resist the "general-purpose / beats-Zstd" overclaim every phase; must
benchmark against the *right* (lossy) baselines; the Cirq layer must stay honest
(no advantage claims).
**Revisit when:** Phase 2 benchmarks land — if TT doesn't beat SVD/PCA even on
structured tensors, **scope down or pivot** (and say so). Let the numbers decide.

## Action Items
1. [ ] Scaffold the repo (Rust workspace + `python/` + `cuda/` + `cirq/` + `docs/`).
2. [ ] Phase 1–2 first; **do not** write CUDA kernels (Phase 3) until Phase 2 proves
       TT beats the SVD/PCA baseline on a real structured tensor.
3. [ ] Every benchmark states the baseline + the data regime (the "X vs Y, lossy,
       structured-tensor" honesty rule). No bare ratios.
4. [ ] Cirq bridge carries a "verification only, no advantage claim" note in its README.

---

### Review discipline
Every phase is reviewed for correctness, honesty (claims match measured numbers + the
right baselines), ecosystem fit, and the gates above. Code that fails a gate does not
advance. The benchmark discipline (`[KNOWN_LIMIT]`, honest baselines) is non-negotiable.
