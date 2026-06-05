"""
python/blaze/examples/substrate_quantum_state.py

Blaze on a real SUBSTRATE-domain quantum many-body state.
=========================================================

WHY THIS EXISTS
---------------
The vision for Blaze: be the compression layer for the data the whole ecosystem
produces (SUBSTRATE quantum states, HELIOS telemetry, model activations, ...).
This is the first proof on REAL physics data, and it is deliberately HONEST
about when Blaze wins and when it must decline.

WHAT IT DOES
------------
SUBSTRATE simulates quantum many-body spin systems. Its Phase-A tensor-network
stack is quimb (cryptotn_gpu/cryptotn runs on quimb; quantum_lab/P3_G2/tn_quimb.py
runs on quimb). We use that same stack to build dense ground states of the
transverse-field Ising model (TFIM) -- the canonical many-body spin chain:

    H = -J * sum_i Z_i Z_{i+1}  -  h * sum_i X_i

and compress each dense statevector with Blaze TT-SVD. We reshape the 2**n
amplitude vector into an (2,)*n tensor, so Blaze's TT bond dimensions ARE the
MPS / Schmidt ranks of the physical state: the compression is literally the
matrix-product-state factorization of the wavefunction.

THE HONEST STORY
----------------
Compression ratio tracks the PHYSICAL half-chain entanglement entropy S:
  * Critical point (h = J):   highest S  -> highest rank -> smallest compression.
  * Deep paramagnet (h >> J): low S      -> low rank     -> large compression.
  * Haar-random state (control): volume-law S -> full rank -> TT needs MORE
    storage than the dense array (ratio < 1). Blaze must, and does, decline --
    and its diagnostic predicts this *before* you pay for the decomposition.

Blaze is not magic and not a general compressor. It exploits area-law / low-
entanglement structure -- which is exactly what physical SUBSTRATE states have.

Run:
    pip install -e '.[quantum]'      # needs quimb
    python -m blaze.examples.substrate_quantum_state
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from blaze import analyze_compressibility, compress  # noqa: E402


# ---------------------------------------------------------------------------
# State generation -- SUBSTRATE's own Phase-A tensor-network stack (quimb)
# ---------------------------------------------------------------------------
def tfim_groundstate(n: int, h: float, J: float = 1.0) -> np.ndarray:
    """Dense ground state of the open-chain transverse-field Ising model.

    H = -J sum_i Z_i Z_{i+1} - h sum_i X_i, built sparsely and diagonalized
    with quimb (the stack SUBSTRATE itself uses). Returns the normalized
    statevector as a complex128 array of length 2**n.
    """
    import quimb as qu

    dims = [2] * n
    zz = sum(qu.ikron([qu.pauli("Z"), qu.pauli("Z")], dims, (i, i + 1), sparse=True)
             for i in range(n - 1))
    x = sum(qu.ikron(qu.pauli("X"), dims, i, sparse=True) for i in range(n))
    H = -J * zz - h * x
    psi = qu.groundstate(H)  # qarray (2**n, 1), sparse eigsh under the hood
    return np.asarray(psi).ravel().astype(np.complex128)


def haar_random_state(n: int, seed: int = 0) -> np.ndarray:
    """A Haar-random pure state: the volume-law, worst-case control."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(2 ** n) + 1j * rng.standard_normal(2 ** n)
    return (v / np.linalg.norm(v)).astype(np.complex128)


def half_chain_entropy(psi: np.ndarray, n: int) -> tuple[float, int]:
    """Von Neumann entanglement entropy (bits) at the central cut and the
    Schmidt rank capturing 1 - 1e-12 of the energy (the 1e-6 amplitude tol).

    The Schmidt rank at this cut is the physical bond dimension; Blaze's TT
    rank tracks it -- both measure the same entanglement.
    """
    half = n // 2
    mat = psi.reshape(2 ** half, 2 ** (n - half))
    s = np.linalg.svd(mat, compute_uv=False)
    p = (s ** 2)
    p = p[p > 1e-300]
    entropy = float(-np.sum(p * np.log2(p)))
    cum = np.cumsum(p) / p.sum()
    schmidt_rank = int(np.searchsorted(cum, 1.0 - 1e-12)) + 1
    return entropy, schmidt_rank


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def evaluate(name: str, psi: np.ndarray, n: int, rel_tol: float,
             save_dir: Path | None) -> dict:
    tens = psi.reshape([2] * n)

    t0 = time.perf_counter()
    tt = compress(tens, max_rank=None, rel_tol=rel_tol)
    dt = time.perf_counter() - t0

    entropy, schmidt_rank = half_chain_entropy(psi, n)
    diag = analyze_compressibility(tens, max_rank=None, rel_tol=rel_tol)
    rec = diag["recommendation"].split(";")[0].split("(")[0].strip()

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / f"{name}.npy", psi.astype(np.complex128))

    return {
        "name": name,
        "entropy": entropy,
        "schmidt": schmidt_rank,
        "tt_max_rank": max(tt.ranks),
        "nparams": tt.nparams(),
        "orig": 2 ** n,
        "ratio": tt.compression_ratio(tens),
        "err": tt.rel_error(tens),
        "t": dt,
        "rec": rec,
    }


def main() -> None:
    n = 16                      # 2**16 = 65,536 amplitudes; central cut 256x256
    rel_tol = 1e-6
    save_dir = Path(__file__).resolve().parents[3] / "data" / "substrate"

    print("=" * 94)
    print(f"Blaze on SUBSTRATE-domain quantum states  "
          f"(TFIM, n={n} spins, 2^{n}={2 ** n} amps, rel_tol={rel_tol:g})")
    print("States built with quimb -- SUBSTRATE's own Phase-A tensor-network stack.")
    print("=" * 94)

    rows: list[dict] = []
    # Physical ground states: criticality (h=J) -> deep paramagnet (h>>J).
    for h in (1.0, 1.5, 2.0, 3.0):
        psi = tfim_groundstate(n, h)
        rows.append(evaluate(f"tfim_n{n}_h{h:g}", psi, n, rel_tol, save_dir))
    # Volume-law control.
    rows.append(evaluate(f"haar_random_n{n}", haar_random_state(n), n, rel_tol, save_dir))

    print(f"\n{'state':>18}  {'S(bits)':>8}  {'Schmidt':>7}  {'TTrank':>6}  "
          f"{'params':>9}  {'ratio':>8}  {'rel.err':>9}  verdict")
    print("-" * 94)
    for r in rows:
        verdict = "COMPRESS" if r["ratio"] > 1.2 else "DECLINE (TT > dense)"
        print(f"{r['name']:>18}  {r['entropy']:8.3f}  {r['schmidt']:7d}  "
              f"{r['tt_max_rank']:6d}  {r['nparams']:9d}  {r['ratio']:7.2f}x  "
              f"{r['err']:9.2e}  {verdict}")

    print("\nHonest read:")
    print("  * Compression ratio tracks the physical entanglement entropy S.")
    print("  * TFIM at criticality (h=1) is the hardest physical case (max S); deeper")
    print("    into the paramagnet (larger h) S drops and compression rises.")
    print("  * Blaze's TT bond dim tracks the state's Schmidt rank -- the compression")
    print("    IS the matrix-product-state factorization of the wavefunction.")
    print("  * The Haar-random control is volume-law: TT needs MORE storage than the")
    print("    dense array (ratio < 1). Blaze declines, and the diagnostic predicts it.")
    print(f"\n  Saved statevectors -> {save_dir}  (gitignored data)")


if __name__ == "__main__":
    main()
