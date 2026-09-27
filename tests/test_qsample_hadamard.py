"""Hadamard on the q-sample cores, not on the probability tensor.

A gate on one mode rewrites that core and leaves the bonds unchanged. On the
6-bit state the float64 Walsh spectrum stays at 1e-12, and hadamard.lyth
matches the f32 formula on both machines. The f32 path does not claim 1e-12.

The ROI q-sample has a bond of 3. A physical slice then exceeds one 256-bit
register, so the machine does not run.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from blaze import Kind, verdict

DESKTOP = Path(__file__).resolve().parents[4]
ROI = DESKTOP / "PR0JECTS" / "RESEARCH" / "QuBLAR" / "build" / "ising_out_roi.txt"
BRANCHES = DESKTOP / "PR0JECTS" / "RESEARCH" / "QuBLAR" / "tools" / "branches_blaze.py"
REL_TOL = 1e-6
FIXTURE_TOL = 1e-12
LANE = 8


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _qsample() -> np.ndarray:
    state = np.zeros((2,) * 6, dtype=np.float64)
    state[(0, 0, 0, 0, 0, 0)] = 0.5
    state[(0, 0, 1, 1, 0, 0)] = 0.5
    return np.sqrt(state)


def _fits_one_register(cores) -> bool:
    for core in cores:
        if core.shape[1] != 2:
            return False
        if int(core.shape[0] * core.shape[2]) > LANE:
            return False
    return True


def test_qsample_hadamard_keeps_its_bonds(tmp_path: Path):
    psi = _qsample()
    answer = verdict(psi, rel_tol=FIXTURE_TOL)
    assert answer.kind is Kind.COMPRESSED
    assert answer.tt is not None
    assert _fits_one_register(answer.tt.cores)
    bonds = [int(core.shape[2]) for core in answer.tt.cores]

    walsh = _load(Path(__file__).resolve().parent / "test_walsh_machines.py", "walsh_qsample")
    dense = walsh._walsh_dense(psi)
    floated = [walsh._hadamard_core(core) for core in answer.tt.cores]
    assert [int(core.shape[2]) for core in floated] == bonds
    assert walsh._rel(answer.tt, floated, dense) <= 1e-12

    n, planes, widths = walsh._pack(answer.tt.cores)
    fixture = tmp_path / "qsample_h.txt"
    walsh._write(fixture, n, planes)
    machine = walsh._machine_cores(answer.tt.cores, walsh._run(fixture), widths)
    assert [int(core.shape[2]) for core in machine] == bonds


def test_roi_qsample_does_not_fit_the_register():
    assert ROI.is_file(), "missing QuBLAR build/ising_out_roi.txt"
    roi = _load(BRANCHES, "branches_blaze_qsample_h")
    _, _, _, h, j = roi.load_roi(ROI)
    posterior = roi.exact_posterior(h, j)
    n = len(h)
    answer = verdict(np.sqrt(posterior).reshape((2,) * n), rel_tol=REL_TOL)
    assert answer.kind is Kind.COMPRESSED
    assert answer.tt is not None
    assert max(answer.tt.ranks) == 3
    assert not _fits_one_register(answer.tt.cores)
