"""Regression test: Blaze on real SUBSTRATE-domain quantum states.

Locks the honest invariants from docs/SUBSTRATE-validation.md:
  * a low-entanglement physical state (deep paramagnet) compresses well,
  * a Haar-random (volume-law) state does NOT (its TT is larger than dense),
  * compression tracks physical entanglement (paramagnet ratio > critical ratio),
  * Blaze's TT bond dim tracks the physical Schmidt rank.

Skips cleanly if quimb (SUBSTRATE's Phase-A TN stack) is not installed.
Uses n = 10 spins for speed (sparse eigsh on a 1024-dim Hamiltonian).
"""

import pytest

pytest.importorskip("quimb")  # SUBSTRATE Phase-A stack; optional dependency

from blaze import compress  # noqa: E402
from blaze.examples.substrate_quantum_state import (  # noqa: E402
    haar_random_state,
    half_chain_entropy,
    tfim_groundstate,
)

N = 10
REL_TOL = 1e-6


def _evaluate(psi):
    tens = psi.reshape([2] * N)
    tt = compress(tens, max_rank=None, rel_tol=REL_TOL)
    _, schmidt = half_chain_entropy(psi, N)
    return {
        "ratio": tt.compression_ratio(tens),
        "err": tt.rel_error(tens),
        "max_rank": max(tt.ranks),
        "schmidt": schmidt,
    }


def test_paramagnet_compresses():
    r = _evaluate(tfim_groundstate(N, h=3.0))
    assert r["ratio"] > 2.0, f"paramagnet should compress, got {r['ratio']:.2f}x"
    assert r["err"] < 1e-5, f"reconstruction error too high: {r['err']:.2e}"
    # TT bond dim tracks the physical Schmidt rank (tolerance distribution -> +1)
    assert r["max_rank"] <= r["schmidt"] + 2, (
        f"TT rank {r['max_rank']} should track Schmidt rank {r['schmidt']}"
    )


def test_haar_random_declines():
    r = _evaluate(haar_random_state(N, seed=0))
    # Volume-law: the TT needs MORE storage than the dense array. Blaze must decline.
    assert r["ratio"] < 1.0, f"Haar state must not compress, got {r['ratio']:.2f}x"
    # essentially full Schmidt rank at the central cut
    assert r["schmidt"] >= 2 ** (N // 2) - 2


def test_compression_tracks_entanglement():
    # Critical (h=J) has more entanglement than the deep paramagnet (h>>J),
    # so it must compress *less*. This is the core honest claim.
    crit = _evaluate(tfim_groundstate(N, h=1.0))
    para = _evaluate(tfim_groundstate(N, h=3.0))
    assert para["ratio"] > crit["ratio"], (
        f"ratio should rise as entanglement falls: "
        f"critical {crit['ratio']:.2f}x, paramagnet {para['ratio']:.2f}x"
    )
