"""
python/blaze/examples/quantum_similarity_search.py

Phase 7 demo — similarity search over quantum states, ENTIRELY in compressed space.
================================================================================

THE IDEA (ADR-0002)
-------------------
Vector-search libraries win by operating on the *compressed* codes, never
decompressing. Blaze does the tensor-network-native version: it stores quantum
states as TT/MPS and compares them with the overlap "zipper" — an O(n·χ³)
contraction that gives the EXACT ⟨ψ|φ⟩ without ever rebuilding the 2ⁿ amplitudes.

WHAT THIS SHOWS
---------------
We build a database of transverse-field Ising (TFIM) ground states across the
phase diagram (a grid of fields h), compress each to a TT, and index them. Then
we query with a *held-out* state at a field NOT in the grid and retrieve the most
similar indexed states by fidelity — all in compressed space, nothing decompressed.

The physics makes it a real test: the ground state varies smoothly with h (away
from the critical point), so the nearest neighbours of a query at h* must be the
grid states with h closest to h*. The index recovers exactly that.

Falls back to a cirq/quimb-free "θ-GHZ" family (cosθ|0…0⟩ + sinθ|1…1⟩) if quimb
is not installed, so the demo always runs.

Run:
    pip install -e '.[quantum]'        # for the TFIM physics path (needs quimb)
    python -m blaze.examples.quantum_similarity_search
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# Windows consoles default to cp1252; the report uses math symbols (chi, ->, ~).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from blaze import TTIndex, compress, fidelity  # noqa: E402
from blaze.overlap import inner  # noqa: E402


def _theta_ghz(n: int, theta: float) -> np.ndarray:
    """cos θ |0…0⟩ + sin θ |1…1⟩ — bond dimension 2, smooth in θ. Always available."""
    psi = np.zeros(2**n, dtype=np.complex128)
    psi[0] = np.cos(theta)
    psi[-1] = np.sin(theta)
    return psi


def _build_database(n: int):
    """Return (label_grid, statevectors, query_label, query_state, kind).

    Uses real TFIM ground states if quimb is present; otherwise the θ-GHZ family.
    """
    try:
        from blaze.examples.substrate_quantum_state import tfim_groundstate

        grid = [0.6, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0]   # transverse field h
        states = [tfim_groundstate(n, h) for h in grid]
        q_label = 1.95                                # held-out field, between 1.8 and 2.2
        query = tfim_groundstate(n, q_label)
        return grid, states, q_label, query, "TFIM ground states (field h)"
    except Exception as exc:  # quimb missing or failed — cirq/quimb-free fallback
        print(f"[info] quimb unavailable ({exc}); using θ-GHZ fallback family.\n")
        grid = list(np.linspace(0.1, 1.4, 7))         # angle θ
        states = [_theta_ghz(n, t) for t in grid]
        q_label = 0.93                                # held-out angle, near 0.9667 grid pt
        query = _theta_ghz(n, q_label)
        return grid, states, q_label, query, "θ-GHZ states (angle θ)"


def main() -> None:
    n = 16
    rel_tol = 1e-8
    grid, states, q_label, query_state, kind = _build_database(n)

    # Compress every database state + the query to TT. From here on: compressed space.
    t0 = time.perf_counter()
    tts = [compress(psi.reshape((2,) * n), rel_tol=rel_tol) for psi in states]
    q_tt = compress(query_state.reshape((2,) * n), rel_tol=rel_tol)
    t_compress = time.perf_counter() - t0

    index = TTIndex(metric="fidelity")
    index.add_many(tts, labels=grid)

    orig_bytes = (2**n) * 16                      # complex128 dense statevector
    tt_bytes = sum(c.nbytes for c in q_tt.cores)

    print("=" * 78)
    print(f"Phase 7 — similarity search in COMPRESSED space ({kind})")
    print(f"n={n} qubits, 2^{n}={2**n} amplitudes/state, {len(grid)} states indexed")
    print(f"each dense state = {orig_bytes/1024:.0f} KiB; one compressed TT ≈ "
          f"{tt_bytes/1024:.1f} KiB ({orig_bytes/tt_bytes:.0f}x smaller)")
    print(f"compressed {len(grid)+1} states in {t_compress*1e3:.1f} ms")
    print("=" * 78)

    # The query (held-out parameter): retrieve nearest by fidelity, no decompression.
    t0 = time.perf_counter()
    top = index.query(q_tt, k=3)
    t_query = time.perf_counter() - t0

    print(f"\nQuery: held-out state at parameter = {q_label}")
    print(f"Top-3 nearest indexed states by fidelity (query took {t_query*1e3:.2f} ms, "
          f"all in compressed space):")
    for rank, (label, score, idx) in enumerate(top, 1):
        print(f"  {rank}.  parameter={label:<5}  fidelity={score:.6f}")

    nearest = min(grid, key=lambda g: abs(g - q_label))
    ok = abs(top[0][0] - nearest) < 1e-9
    print(f"\nExpected nearest by parameter proximity: {nearest}  "
          f"-> retrieved: {top[0][0]}   [{'OK' if ok else 'MISMATCH'}]")

    # Cross-fidelity matrix (compressed space) — the smooth structure + criticality.
    print("\nPairwise fidelity matrix (computed on the TTs directly, never decompressed):")
    header = "        " + "".join(f"{g:>7.2f}" for g in grid)
    print(header)
    for i, gi in enumerate(grid):
        row = f"{gi:>6.2f}  "
        for j, gj in enumerate(grid):
            row += f"{fidelity(tts[i], tts[j]):>7.3f}"
        print(row)

    print("\nHonest read:")
    print("  * Every number above is an EXACT overlap (fp-precise), computed by the")
    print("    O(n·χ³) zipper — no 2^n amplitude vector is ever reconstructed.")
    print("  * Nearest-neighbour in compressed space tracks the physical parameter:")
    print("    states close in h (or θ) have fidelity → 1; far ones decohere toward 0.")
    print("  * [KNOWN_LIMIT] This is cheap because these states are low-χ. A volume-law")
    print("    (Haar) database would have χ ~ 2^(n/2): the overlap costs as much as the")
    print("    dense product. The index is exact and fast ONLY for TT-structured data.")


if __name__ == "__main__":
    main()
