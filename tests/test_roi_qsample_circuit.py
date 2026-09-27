"""The ROI q-sample, prepared where χ = 3 fits, and not on ZIPPER2.

√p of the 20 ghost bits keeps a bond of 3 at rel_tol = 1e-6. Phase 5 can
prepare that MPS. The fidelity of the circuit against the MPS, and the
ancilla check, use the Phase 5 bar 1 − 1e-9. The fidelity of the MPS
against the exact amplitudes uses the verdict budget, 1 − 1e-6.

RESULTS-phase6: the MAP branch holds 0.974 and three states hold 99 %.
Those floors are not moved. max_rank = 2 stays undecided, so the 256-bit
word does not run.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cirq")

from blaze import Kind, verdict  # noqa: E402
from blaze.cirq import verify_mps_circuit  # noqa: E402

DESKTOP = Path(__file__).resolve().parents[4]
ROI = DESKTOP / "PR0JECTS" / "RESEARCH" / "QuBLAR" / "build" / "ising_out_roi.txt"
BRANCHES = DESKTOP / "PR0JECTS" / "RESEARCH" / "QuBLAR" / "tools" / "branches_blaze.py"
REL_TOL = 1e-6
PHASE5 = 1.0 - 1e-9
MAP_FLOOR = 0.974
TOP3_FLOOR = 0.99


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_roi_qsample_is_prepared_off_the_word():
    assert ROI.is_file(), "missing QuBLAR build/ising_out_roi.txt"
    roi = _load(BRANCHES, "branches_blaze_roi_circuit")
    _, _, _, h, j = roi.load_roi(ROI)
    posterior = roi.exact_posterior(h, j)
    assert np.min(posterior) >= 0.0
    psi = np.sqrt(posterior)
    n = len(h)
    state = verdict(psi.reshape((2,) * n), rel_tol=REL_TOL)
    assert state.kind is Kind.COMPRESSED
    assert state.tt is not None
    assert max(state.tt.ranks) == 3

    refused = verdict(psi.reshape((2,) * n), rel_tol=REL_TOL, max_rank=2)
    assert refused.kind is Kind.UNDECIDED
    assert refused.tt is None

    recon = np.asarray(state.tt.reconstruct()).ravel()
    numer = abs(np.vdot(psi, recon)) ** 2
    denom = float(np.vdot(psi, psi).real * np.vdot(recon, recon).real)
    assert float(numer / denom) >= 1.0 - REL_TOL

    checked = verify_mps_circuit(state.tt)
    assert checked["fidelity"] >= PHASE5
    assert checked["ancilla_disentangle"] >= PHASE5
    assert checked["max_bond"] == 3
    assert checked["ancilla_qubits"] == 2
    assert checked["n_qubits"] == 20

    heavy = np.argpartition(posterior, -3)[-3:]
    assert float(posterior.max()) >= MAP_FLOOR
    assert float(posterior[heavy].sum()) >= TOP3_FLOOR
    born = np.abs(recon) ** 2
    born /= float(born.sum())
    assert int(np.argmax(born)) == int(np.argmax(posterior))
    assert float(born[heavy].sum()) >= TOP3_FLOOR
