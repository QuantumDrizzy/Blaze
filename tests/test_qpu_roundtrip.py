"""QPU round trip for the 6-bit q-sample.

The state vector is √p. The gate is hadamard.lyth on the top qubit, n = 32
per half, on the host, on PTX and on the emulator. If those bits miss the
closed form, the gate does not continue into Blaze.

Both directions use the Phase 5 fidelity bar, 1 − 1e-9, and the fixture
tolerance 1e-12. The ROI q-sample is not loaded: it does not fit χ = 2.
"""

from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path

import numpy as np

from blaze import Kind, verdict

PHASE5_FIDELITY = 1.0 - 1e-9
FIXTURE_TOL = 1e-12
S = np.float32(2.0**-0.5)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _bits(values: np.ndarray) -> list[int]:
    return [struct.unpack("<I", np.float32(value).tobytes())[0] for value in values]


def _f32(bits: int) -> np.float32:
    return np.float32(struct.unpack("<f", struct.pack("<I", bits))[0])


def _fidelity(amplitudes: np.ndarray, tt) -> float:
    recon = np.asarray(tt.reconstruct()).ravel()
    state = np.asarray(amplitudes).ravel()
    numer = abs(np.vdot(state, recon)) ** 2
    denom = float(np.vdot(state, state).real * np.vdot(recon, recon).real)
    return float(numer / denom)


def _qsample() -> np.ndarray:
    p = np.zeros((2,) * 6, dtype=np.float64)
    p[(0, 0, 0, 0, 0, 0)] = 0.5
    p[(0, 0, 1, 1, 0, 0)] = 0.5
    return np.sqrt(p)


def _closed_or_none(q0: np.ndarray, q1: np.ndarray, q0i: list[int], q1i: list[int], p0: np.ndarray, p1: np.ndarray):
    """None means the gate missed its closed form and Blaze is not called."""
    closed0 = S * (p0.astype(np.float32) + p1.astype(np.float32))
    closed1 = S * (p0.astype(np.float32) - p1.astype(np.float32))
    n = int(p0.size)
    if q0.tobytes() != np.asarray(closed0, dtype=np.float32).tobytes():
        return None
    if q1.tobytes() != np.asarray(closed1, dtype=np.float32).tobytes():
        return None
    if q0i != [0] * n or q1i != [0] * n:
        return None
    return np.stack([q0, q1]).astype(np.float64).reshape((2,) * 6)


def _apply_or_refuse(p0: np.ndarray, p1: np.ndarray, tmp_path: Path):
    walsh = _load(Path(__file__).resolve().parent / "test_walsh_machines.py", "walsh_qpu")
    n = int(p0.size)
    assert n == 32 and p1.size == n
    planes = [_bits(p0), [0] * n, _bits(p1), [0] * n]
    fixture = tmp_path / "qpu.txt"
    walsh._write(fixture, n, planes)
    got = walsh._run(fixture)
    q0 = np.array([_f32(bits) for bits in got["q0r"]], dtype=np.float32)
    q1 = np.array([_f32(bits) for bits in got["q1r"]], dtype=np.float32)
    return _closed_or_none(q0, q1, got["q0i"], got["q1i"], p0, p1)


def test_qsample_round_trip_through_the_qpu(tmp_path: Path):
    psi = _qsample()
    outbound = verdict(psi, rel_tol=FIXTURE_TOL)
    assert outbound.kind is Kind.COMPRESSED
    assert outbound.tt is not None
    assert _fidelity(psi, outbound.tt) >= PHASE5_FIDELITY

    halves = psi.reshape(2, 32)
    phi = _apply_or_refuse(halves[0], halves[1], tmp_path)
    assert phi is not None

    inbound = verdict(phi, rel_tol=FIXTURE_TOL)
    assert inbound.kind is Kind.COMPRESSED
    assert inbound.tt is not None
    assert inbound.tt.ranks == [1, 1, 1, 2, 1, 1, 1]
    assert _fidelity(phi, inbound.tt) >= PHASE5_FIDELITY


def test_a_miss_does_not_enter_blaze():
    psi = _qsample()
    halves = psi.reshape(2, 32)
    q0 = halves[0].astype(np.float32).copy()
    q1 = halves[1].astype(np.float32).copy()
    q0[0] = np.float32(q0[0] + np.float32(1.0))
    assert _closed_or_none(q0, q1, [0] * 32, [0] * 32, halves[0], halves[1]) is None
