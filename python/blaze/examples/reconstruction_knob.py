"""
python/blaze/examples/reconstruction_knob.py

What Blaze does to the data: before -> after, and the error<->rank knob.

Compress a physical SUBSTRATE-domain state and a Haar-random control, reconstruct,
and show (a) storage before/after, (b) what is preserved (wavefunction overlap +
a physical observable), (c) the controllable, monotone error<->rank tradeoff.

This is the Phase-4 characterization (docs/PHASE4-results.md): approximate
reconstruction with an honest, monotone error<->rank curve.

Run:
    pip install -e '.[quantum]'
    python -m blaze.examples.reconstruction_knob
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from blaze import compress  # noqa: E402
from blaze.examples.substrate_quantum_state import (  # noqa: E402
    haar_random_state,
    tfim_groundstate,
)

N = 16
ITEM_BYTES = 16  # complex128


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel(), b.ravel()
    return abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))


def transverse_magnetization(psi: np.ndarray, n: int = N) -> float:
    """(1/n) sum_i <X_i>: X_i swaps |0>,|1> on axis i (a flip of that axis)."""
    psi = psi.reshape([2] * n)
    return sum(np.real(np.vdot(psi.ravel(), np.flip(psi, axis=i).ravel()))
               for i in range(n)) / n


def before_after(name: str, psi: np.ndarray, rel_tol: float = 1e-6) -> None:
    tens = psi.reshape([2] * N)
    tt = compress(tens, max_rank=None, rel_tol=rel_tol)
    recon = tt.reconstruct()
    before = tens.size * ITEM_BYTES
    after = tt.nparams() * ITEM_BYTES  # TT-core payload (= .blz minus 279-byte header)
    print(f"\n[{name}]")
    print(f"  storage    before {before:>10,} B  ->  after {after:>10,} B   ({before / after:6.2f}x)")
    print(f"  overlap    |<orig|recon>|             = {overlap(psi, recon):.12f}")
    print(f"  observable <Mx>/n  before -> after    = {transverse_magnetization(psi):.8f}"
          f" -> {transverse_magnetization(recon):.8f}")


def knob(psi: np.ndarray, ranks=(1, 2, 3, 4, 6, 8, 12, 16)) -> None:
    tens = psi.reshape([2] * N)
    print(f"\n  {'max_rank':>8}  {'nparams':>8}  {'ratio':>7}  {'rel_error':>11}")
    prev = None
    for mr in ranks:
        tt = compress(tens, max_rank=mr, rel_tol=1e-12)
        err = tt.rel_error(tens)
        ratio = tens.size / tt.nparams()
        flag = "" if prev is None else (" monotone" if err <= prev + 1e-12 else " NONMONOTONE!")
        print(f"  {mr:>8}  {tt.nparams():>8}  {ratio:6.2f}x  {err:11.3e}{flag}")
        prev = err


def main() -> None:
    print("=" * 72)
    print("WHAT BLAZE DOES: before -> after (compress -> reconstruct)")
    print("=" * 72)
    before_after("paramagnet h=3 (physical)", tfim_groundstate(N, 3.0))
    before_after("Haar-random (control)", haar_random_state(N))

    print("\n" + "=" * 72)
    print("THE KNOB (Phase 4): TFIM critical h=1 — controllable, monotone")
    print("=" * 72)
    knob(tfim_groundstate(N, 1.0))


if __name__ == "__main__":
    main()
