"""Minimal roundtrip and golden property tests for Phase 1 TT core.

These run without optional deps.
"""

import numpy as np
import pytest

from blaze import TT, compress, tt_svd


def test_low_rank_roundtrip_exact():
    rng = np.random.default_rng(42)
    # Build exact small TT
    cores = [
        rng.standard_normal((1, 4, 3)).astype(np.float32),
        rng.standard_normal((3, 5, 2)).astype(np.float32),
        rng.standard_normal((2, 3, 1)).astype(np.float32),
    ]
    tt = TT(cores=cores, shape=(4, 5, 3), singular_values=[])
    exact = tt.reconstruct()
    noisy = exact + 1e-6 * rng.standard_normal(exact.shape).astype(np.float32)

    tt2 = compress(noisy, max_rank=5, rel_tol=1e-5)
    err = tt2.rel_error(noisy)
    assert err < 1e-4, f"roundtrip error too high: {err}"
    assert tt2.ranks == [1, 3, 2, 1] or max(tt2.ranks[1:-1]) <= 3  # roughly recovers


def test_diagnostics_runs_and_reports():
    t = np.random.randn(6, 5, 7).astype(np.float32) * 0.1 + 0.5
    info = __import__("blaze.diagnostics", fromlist=["analyze_compressibility"]).analyze_compressibility(
        t, max_rank=8, rel_tol=1e-2
    )
    assert "effective_ranks_at_tol" in info
    assert len(info["effective_ranks_at_tol"]) == 2  # ndim-1 bonds


def test_complex128_path():
    rng = np.random.default_rng(0)
    shape = (3, 4, 3)
    # low rank complex
    c0 = (rng.standard_normal((1, 3, 2)) + 1j * rng.standard_normal((1, 3, 2))).astype(np.complex128)
    c1 = (rng.standard_normal((2, 4, 2)) + 1j * rng.standard_normal((2, 4, 2))).astype(np.complex128)
    c2 = (rng.standard_normal((2, 3, 1)) + 1j * rng.standard_normal((2, 3, 1))).astype(np.complex128)
    tt = TT([c0, c1, c2], shape=shape, singular_values=[])
    exact = tt.reconstruct()
    tt2 = compress(exact + 1e-5j * rng.standard_normal(shape), max_rank=3, rel_tol=1e-4)
    assert tt2.rel_error(exact) < 1e-3
    assert np.iscomplexobj(tt2.cores[0])
