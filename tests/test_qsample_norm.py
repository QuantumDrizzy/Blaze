"""Q-sample that fits χ = 2, contracted as ⟨ψ|ψ⟩ on both machines.

The state is √p of the 6-bit fixture. Its bonds are 2, so ZIPPER2 runs.
Ket and bra are the same packed core: the overlap, not the probability marginal.
The ROI q-sample stays out; test_sqrt_p_ranks.py already records that max_rank
2 is undecided there.

Fidelity against the dense amplitudes uses the Phase 5 bar, 1 − 1e-9. The
machine check is bit-exact with the f32 zipper. The norm is that contraction,
not a second tolerance.
"""

from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path

import numpy as np

from blaze import Kind, fidelity, quantize_tt, verdict

PHASE5_FIDELITY = 1.0 - 1e-9
FIXTURE_TOL = 1e-12


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _dense_fidelity(amplitudes: np.ndarray, tt) -> float:
    recon = np.asarray(tt.reconstruct()).ravel()
    state = np.asarray(amplitudes).ravel()
    numer = abs(np.vdot(state, recon)) ** 2
    denom = float(np.vdot(state, state).real * np.vdot(recon, recon).real)
    return float(numer / denom)


def test_six_bit_qsample_overlaps_on_both_machines(tmp_path: Path):
    p = np.zeros((2,) * 6, dtype=np.float64)
    p[(0, 0, 0, 0, 0, 0)] = 0.5
    p[(0, 0, 1, 1, 0, 0)] = 0.5
    psi = np.sqrt(p)
    answer = verdict(psi, rel_tol=FIXTURE_TOL)
    assert answer.kind is Kind.COMPRESSED
    assert answer.tt is not None
    assert max(answer.tt.ranks) <= 2
    assert _dense_fidelity(psi, answer.tt) >= PHASE5_FIDELITY

    qgpu = _load(Path(__file__).resolve().parent / "test_qgpu_zipper.py", "qgpu_qsample")
    gate = _load(Path(__file__).resolve().parent / "test_ghost_bits_gate.py", "ghost_qsample")
    deq = quantize_tt(answer.tt, bits=8, granularity="per_bond").dequantize()
    assert fidelity(deq, answer.tt) >= PHASE5_FIDELITY
    assert _dense_fidelity(psi, deq) >= PHASE5_FIDELITY

    packed = [qgpu._lanes_from_core(qgpu._pad(np.asarray(core))) for core in deq.cores]
    chains = [[(list(lanes), list(lanes)) for lanes in packed]]
    rows = gate._run_machines(chains, tmp_path)
    assert len(rows) == 1

    transfer = np.zeros((2, 2), dtype=np.complex64)
    transfer[0, 0] = np.float32(1.0)
    for lanes in packed:
        transfer = qgpu._zipper(transfer, qgpu._unpack(lanes), qgpu._unpack(lanes))
    got_re = np.float32(struct.unpack("<f", struct.pack("<I", rows[0][0]))[0])
    got_im = np.float32(struct.unpack("<f", struct.pack("<I", rows[0][1]))[0])
    assert got_re == np.float32(transfer[0, 0].real)
    assert got_im == np.float32(transfer[0, 0].imag)
    assert rows[0][1] == 0
