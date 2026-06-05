"""Phase 4 gate: approximate reconstruction is controllable and honest.

The relative Frobenius error must decrease (never increase) as the bond-rank cap
grows -- the lossy knob -- and reconstruction at sufficient rank must be faithful.

Tested on a synthetic complex tensor with a known moderate TT rank plus a small
noise floor, so it has no physics dependencies and always runs.
"""

import numpy as np

from blaze import TT, compress


def _structured_tensor(seed: int = 0) -> np.ndarray:
    """A complex tensor with a known moderate TT rank + small noise floor."""
    rng = np.random.default_rng(seed)

    def c(shape):
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    cores = [c((1, 4, 3)), c((3, 5, 5)), c((5, 6, 4)), c((4, 4, 1))]
    exact = TT(cores=cores, shape=(4, 5, 6, 4), singular_values=[]).reconstruct()
    noise = 1e-3 * (rng.standard_normal(exact.shape)
                    + 1j * rng.standard_normal(exact.shape))
    return exact + noise


def test_error_rank_curve_is_monotone():
    t = _structured_tensor()
    errs = [compress(t, max_rank=r, rel_tol=1e-14).rel_error(t) for r in range(1, 7)]
    # non-increasing across the whole sweep (the controllable knob)
    for lo, hi in zip(errs[1:], errs[:-1]):
        assert lo <= hi + 1e-9, f"error must not grow with rank: {errs}"
    # the knob actually has range: more rank really cuts the error
    assert errs[-1] < errs[0] * 0.1, f"higher rank should reduce error: {errs}"


def test_reconstruction_faithful_at_full_rank():
    t = _structured_tensor()
    tt = compress(t, max_rank=None, rel_tol=1e-12)
    # at full rank the reconstruction reaches the ~1e-3 noise floor
    assert tt.rel_error(t) < 5e-3
    recon = tt.reconstruct()
    assert recon.shape == t.shape
    assert np.iscomplexobj(recon)
