"""Ranks of √p, measured at the tolerance already declared for p.

|ψ⟩ = Σ √p(x) |x⟩. Nothing here assumes those bonds equal the bonds of p.
The 6-bit fixture is a scalar multiple, so the bonds match for that reason.
The ROI is not a scalar multiple. At rel_tol = 1e-6 its q-sample keeps a bond
of 3, and max_rank = 2 is undecided: χ = 2 does not hold it, so ZIPPER2 does
not run. The tolerance is the one the ghost-bit gate already used.
"""

from __future__ import annotations

import os
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from blaze import Kind, verdict

# The sibling checkouts (QuBLAR, LYTH): $BLAZE_ECOSYSTEM, else the folder that holds them here.
DESKTOP = Path(os.environ.get("BLAZE_ECOSYSTEM", Path(__file__).resolve().parents[4]))
ROI = DESKTOP / "PR0JECTS" / "RESEARCH" / "QuBLAR" / "build" / "ising_out_roi.txt"
BRANCHES = DESKTOP / "PR0JECTS" / "RESEARCH" / "QuBLAR" / "tools" / "branches_blaze.py"
REL_TOL = 1e-6
FIXTURE_TOL = 1e-12


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _amplitude(p: np.ndarray) -> np.ndarray:
    assert np.min(p) >= 0.0
    psi = np.sqrt(p)
    assert abs(float(np.dot(psi, psi)) - 1.0) <= 1e-12
    return psi


def test_equal_nonzero_weights_keep_the_same_bonds():
    p = np.zeros((2,) * 6, dtype=np.float64)
    p[(0, 0, 0, 0, 0, 0)] = 0.5
    p[(0, 0, 1, 1, 0, 0)] = 0.5
    psi = _amplitude(p.ravel()).reshape(p.shape)
    nonzero = p > 0.0
    ratio = psi[nonzero] / p[nonzero]
    assert np.allclose(ratio, ratio[0])
    assert np.all(psi[~nonzero] == 0.0)

    posterior = verdict(p, rel_tol=FIXTURE_TOL)
    state = verdict(psi, rel_tol=FIXTURE_TOL)
    assert posterior.kind is Kind.COMPRESSED
    assert state.kind is Kind.COMPRESSED
    assert posterior.tt is not None and state.tt is not None
    assert posterior.tt.ranks == [1, 1, 1, 2, 1, 1, 1]
    assert state.tt.ranks == posterior.tt.ranks


@pytest.mark.ecosystem
def test_roi_sqrt_p_is_a_different_train():
    assert ROI.is_file(), "missing QuBLAR build/ising_out_roi.txt"
    roi = _load(BRANCHES, "branches_blaze_sqrt")
    _, _, _, h, j = roi.load_roi(ROI)
    p = roi.exact_posterior(h, j)
    psi = _amplitude(p)
    nonzero = p > 0.0
    ratio = psi[nonzero] / p[nonzero]
    assert not np.allclose(ratio, ratio[0])

    n = len(h)
    posterior = verdict(p.reshape((2,) * n), rel_tol=REL_TOL)
    state = verdict(psi.reshape((2,) * n), rel_tol=REL_TOL)
    assert posterior.kind is Kind.COMPRESSED
    assert state.kind is Kind.COMPRESSED
    assert posterior.tt is not None and state.tt is not None
    assert posterior.measured_rel_error is not None
    assert state.measured_rel_error is not None
    assert posterior.measured_rel_error <= REL_TOL
    assert state.measured_rel_error <= REL_TOL

    # p stays inside χ = 2. √p does not: three cuts keep a third singular value
    # whose tail sits far above the unfolding budget.
    assert posterior.tt.ranks == [1, 2, *([1] * 19)]
    assert max(state.tt.ranks) == 3
    assert state.tt.ranks[2:5] == [3, 3, 3]
    assert state.tt.ranks != posterior.tt.ranks

    capped = verdict(psi.reshape((2,) * n), rel_tol=REL_TOL, max_rank=2)
    assert capped.kind is Kind.UNDECIDED
    assert capped.tt is None
    assert capped.cap_rank == 2
