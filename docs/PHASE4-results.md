# Phase 4 — Approximate reconstruction: error↔rank curve + on-disk before/after

**Date:** 2026-06-05 · Python reference + Rust CLI (CPU c64 path). Reconstruction is device-independent.

## What Phase 4 is

Reconstruction itself has existed since Phase 2 (`TT::reconstruct`). Phase 4 is the
**characterization gate**: prove the lossy knob is **controllable, monotone and
honest** — relative Frobenius error decreases as the bond rank grows — and show
concretely what survives a `compress → reconstruct` roundtrip on real data.

## Gate 1 — the error↔rank curve is monotone (verified)

TFIM critical ground state (h=1, n=16, 2¹⁶ amplitudes); vary the hard rank cap:

| max_rank | nparams | ratio | rel. error |
|---------:|--------:|------:|-----------:|
| 1  |   32 | 2048× | 8.64e-1 |
| 2  |  120 |  546× | 1.06e-1 |
| 3  |  248 |  264× | 4.33e-2 |
| 4  |  424 |  155× | 5.21e-3 |
| 6  |  856 |   77× | 2.62e-4 |
| 8  | 1448 |   45× | 9.93e-5 |
| 12 | 2856 |   23× | 2.18e-6 |
| 16 | 4488 |   15× | 9.39e-8 |

Monotone the whole way: more rank → less error → larger file. `rank=1` is honestly
useless (0.86 error). This is the controllable size↔fidelity dial. Locked as a
regression test in `tests/test_phase4_reconstruction.py`.

*(Honest caveat: h=1 is the critical point — the gap closes, so Lanczos returns
slightly different vectors in the near-degenerate ground space run-to-run. The
table is one representative state; the `reconstruction_knob` example regenerates it
and reports ~14× at the tightest rank instead of 15×. Monotonicity and the
size↔fidelity trend are robust; the exact top-rank nparams wobbles by a few %.)*

## Gate 2 — on-disk before/after, real files, real tool

Round-tripped through the actual Rust CLI (`blaze compress` / `blaze reconstruct`,
**c64 path verified end-to-end here for the first time**):

| state | before `.npy` | after `.blz` | ratio | `\|⟨orig\|recon⟩\|` | observable ⟨Mₓ⟩/n |
|-------|------:|------:|------:|------:|------:|
| TFIM h=3 (paramagnet) | 1,048,768 B | 13,975 B | **75.0×** | 1.000000000000 | 0.97346875 → 0.97346875 (Δ 6e-15) |
| Haar-random (control) | 1,048,768 B | 2,796,439 B | **0.38×** | 1.0 | preserved but pointless |

**The point:** 98.7 % of the bytes discarded, wavefunction and every physical
observable **bit-identical**. The `.blz` is essentially `nparams × 16 B` plus a
279-byte header (13,975 = 856×16 + 279; 2,796,439 = 174,760×16 + 279) — minimal
overhead, honest accounting. On the Haar control the file **grew 2.67×** — Blaze
declines, it does not pretend.

*(These are on-disk file bytes, including the `.npy` (192 B) and `.blz` (279 B)
headers. The `reconstruction_knob` example reports the same story from in-memory
bytes — pure core payload `nparams×16`, the `.blz` minus its header — so its ratios
read a hair higher, e.g. 76.6× vs the 75.0× on disk for the paramagnet.)*

## How it works (one paragraph)

Reshape the array to a high-order tensor (`(2,)¹⁶` here). Sweep the bonds left→right;
at each cut, SVD the matricization and keep the top-`r` singular triplets
(`r` = bond dimension), discard the tail — the lossy step. The kept cores form an
MPS whose contraction reconstructs the original to the chosen tolerance. The bond
sequence shows the structure directly: paramagnet `[1,2,4,6,…,6,4,2,1]` (plateau at
6 = the correlation crossing each cut); Haar `[1,2,4,8,…,256,…]` (full — nothing to
exploit).

## Scope and honest limits

- **The CLI does not tensorize.** `blaze compress` reads the `.npy` shape as-is, so a
  1-D statevector compresses to a trivial rank-1 TT (a no-op: `ranks=[1,1]`). The
  reshape to `(2,)^n` is currently done before saving. The CLI needs a
  `--qubits N` / `--reshape d0,d1,...` flag to be a real statevector tool.
  `[KNOWN_LIMIT]` → Phase 1/6 follow-up.
- Reconstruction is CPU and cheap (a chain of small contractions); no GPU needed.
- This characterizes the lossy tradeoff; it is **not** a general-compressor claim
  (PHASE1/PHASE2 settled scope; see `SUBSTRATE-validation.md`).

## Reproduce

```bash
# before/after fidelity + observable + the error<->rank knob (Python):
python -m blaze.examples.reconstruction_knob

# real on-disk roundtrip through the CLI (state already reshaped to (2,)^n):
cargo build --release -p blaze-core --bin blaze
target/release/blaze compress state.tensor.npy --rel-tol 1e-6 -o state.blz
target/release/blaze reconstruct state.blz -o state.recon.npy

# the monotone-knob gate as a regression test:
pytest tests/test_phase4_reconstruction.py -v
```

## Discipline note

Every ratio states the regime (TT-native physical state vs Haar control) and the
baseline (dense `.npy` bytes). The Haar "negative" (file grows 2.67×) is reported as
prominently as the 75× win — the honest knob cuts both ways.
