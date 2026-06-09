"""
python/blaze/examples/quantize_sweep.py

Phase 8 demo — second-stage core quantization on real SUBSTRATE quantum states.
================================================================================

Stage 1 (TT-SVD) compresses STRUCTURE: rank truncation gives ratio R_tt.
Stage 2 (this) compresses the residual NUMERIC CONTENT of the cores: each entry
becomes a b-bit code + a scale, multiplying the ratio by up to 16B/(b/8·2) while
ADDING a measured, composable error.

For each TFIM ground state we report, for b ∈ {8, 6, 4} bits:
  * R_tt        — TT-only ratio vs the dense statevector,
  * R_total     — end-to-end ratio after quantization,
  * x_extra     — the extra factor quantization buys on top of the TT,
  * fidelity    — |⟨ψ_tt | ψ_dequant⟩|² measured with the Phase 7 overlap
                  (in compressed space — the two phases compose).

Honest: fidelity is reported next to every ratio. Aggressive bit widths buy more
ratio and cost more fidelity; the user picks the point. Falls back to a θ-GHZ
family if quimb is absent.

Run:
    pip install -e '.[quantum]'
    python -m blaze.examples.quantize_sweep
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from blaze import compress, fidelity, quantize_tt  # noqa: E402


def _states(n: int):
    """(labels, statevectors). Real TFIM ground states if quimb is present."""
    try:
        from blaze.examples.substrate_quantum_state import tfim_groundstate

        labels = ["TFIM h=1.0 (critical)", "TFIM h=1.5", "TFIM h=2.0", "TFIM h=3.0 (param.)"]
        states = [tfim_groundstate(n, h) for h in (1.0, 1.5, 2.0, 3.0)]
        return labels, states, "TFIM ground states (quimb)"
    except Exception as exc:
        print(f"[info] quimb unavailable ({exc}); θ-GHZ fallback.\n")
        def theta_ghz(theta):
            psi = np.zeros(2**n, dtype=np.complex128)
            psi[0], psi[-1] = np.cos(theta), np.sin(theta)
            return psi
        labels = [f"theta-GHZ θ={t:.2f}" for t in (0.2, 0.5, 0.9, 1.2)]
        return labels, [theta_ghz(t) for t in (0.2, 0.5, 0.9, 1.2)], "θ-GHZ states"


def main() -> None:
    n = 16
    # 1e-6 matches the validated SUBSTRATE run: TT ranks track PHYSICAL entanglement.
    # (Tighter tolerances start capturing the eigensolver's numerical-noise tail as
    # spurious bond dimension — an eigsh artifact, not real structure.)
    rel_tol = 1e-6
    labels, states, kind = _states(n)
    dense_bytes = (2**n) * 16  # complex128

    print("=" * 96)
    print(f"Phase 8 — TT-rank + core quantization, composed.  {kind}, n={n} "
          f"(dense = {dense_bytes//1024} KiB/state)")
    print("=" * 96)
    print(f"{'state':>22}  {'chi':>3}  {'R_tt':>7}  {'bits':>4}  "
          f"{'R_total':>8}  {'x_extra':>7}  {'fidelity':>10}")
    print("-" * 96)

    for label, psi in zip(labels, states):
        tt = compress(psi.reshape((2,) * n), rel_tol=rel_tol)
        tt_bytes = sum(c.nbytes for c in tt.cores)
        r_tt = dense_bytes / tt_bytes
        chi = max(tt.ranks)
        first = True
        for bits in (8, 6, 4):
            q = quantize_tt(tt, bits=bits, granularity="per_bond")
            r_total = q.compression_ratio_over_dense(dense_bytes)
            x_extra = tt_bytes / q.nbytes()
            fid = fidelity(tt, q.dequantize())  # Phase 7 overlap, compressed space
            tag = f"{label:>22}  {chi:>3d}  {r_tt:>6.1f}x" if first else " " * 38
            print(f"{tag}  {bits:>4d}  {r_total:>7.1f}x  {x_extra:>6.2f}x  {fid:>10.6f}")
            first = False
        print("-" * 96)

    print("\nHonest read:")
    print("  * R_total = R_tt × x_extra: the two compression stages multiply.")
    print("  * x_extra approaches 16B→2B = 8x at int8 (less on tiny-χ cores, where the")
    print("    per-bond scale metadata is non-negligible — [KNOWN_LIMIT]).")
    print("  * Fidelity is measured (Phase 7 overlap), never assumed. int8 stays at")
    print("    ~1; 4-bit buys ~2x more ratio at a visible fidelity cost — you choose.")
    print("  * Lossy-on-lossy, always vs the dense original — never vs a lossless codec.")


if __name__ == "__main__":
    main()
