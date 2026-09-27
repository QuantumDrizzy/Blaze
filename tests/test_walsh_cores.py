"""Float64 half of walsh_on_cores.

H on each TT core, ranks unchanged, Walsh spectrum equal to the dense check.
The bar here is 1e-12. The f32 kernel on both machines is test_walsh_machines.py.
"""

import numpy as np

from blaze import compress

S = 2.0 ** -0.5


def _walsh_dense(p: np.ndarray) -> np.ndarray:
    out = np.asarray(p, dtype=np.float64)
    for axis in range(out.ndim):
        moved = np.moveaxis(out, axis, 0)
        lo, hi = moved[0], moved[1]
        mixed = np.stack((S * (lo + hi), S * (lo - hi)), axis=0)
        out = np.moveaxis(mixed, 0, axis)
    return out


def _hadamard_core(core: np.ndarray) -> np.ndarray:
    lo, hi = core[:, 0, :], core[:, 1, :]
    out = np.empty_like(core)
    out[:, 0, :] = S * (lo + hi)
    out[:, 1, :] = S * (lo - hi)
    return out


def test_two_correlated_bits_keep_their_cut():
    # 000000 and 001100, equal weight. Rank 2 only on the cut between those bits.
    p = np.zeros((2,) * 6, dtype=np.float64)
    p[(0, 0, 0, 0, 0, 0)] = 0.5
    p[(0, 0, 1, 1, 0, 0)] = 0.5
    tt = compress(p, rel_tol=1e-12)
    bonds = tt.ranks[1:-1]
    assert bonds[2] == 2
    assert bonds[:2] == [1, 1]
    assert bonds[3:] == [1, 1]

    flipped = [_hadamard_core(core) for core in tt.cores]
    assert [int(core.shape[2]) for core in flipped] == [int(core.shape[2]) for core in tt.cores]
    got = tt.cores
    tt.cores = flipped
    try:
        recon = tt.reconstruct()
    finally:
        tt.cores = got
    dense = _walsh_dense(p)
    err = np.linalg.norm((recon - dense).ravel()) / np.linalg.norm(dense.ravel())
    assert err <= 1e-12
