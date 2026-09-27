"""The 6-bit q-sample, prepared as a circuit and measured.

Phase 5 builds the state-preparation circuit from the MPS. The fidelity bar
and the ancilla check are the ones already in test_phase5_mps_circuit.py:
1 − 1e-9. A computational-basis measurement may only land on the two branches
of p. The ROI q-sample is not prepared: it does not fit χ = 2.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cirq")

from blaze import Kind, verdict  # noqa: E402
from blaze.cirq import mps_to_circuit, sample_from_tt, verify_mps_circuit  # noqa: E402

import cirq  # noqa: E402

PHASE5 = 1.0 - 1e-9
FIXTURE_TOL = 1e-12
BRANCHES = (0, 0b001100)


def _qsample() -> np.ndarray:
    p = np.zeros((2,) * 6, dtype=np.float64)
    p[(0, 0, 0, 0, 0, 0)] = 0.5
    p[(0, 0, 1, 1, 0, 0)] = 0.5
    return np.sqrt(p)


def _prepared_probabilities(tt):
    circuit, ancilla, physical, info = mps_to_circuit(tt)
    order = ancilla + physical
    full = cirq.Simulator(dtype=np.complex128).simulate(circuit, qubit_order=order).final_state_vector
    n = info["n_qubits"]
    m = info["ancilla_qubits"]
    physical_state = np.asarray(full).reshape(2**m, 2**n)[0]
    weights = np.abs(physical_state) ** 2
    weights = weights / (float(weights.sum()) + 1e-30)
    return weights, info


def test_preparation_reproduces_the_two_branches():
    psi = _qsample()
    answer = verdict(psi, rel_tol=FIXTURE_TOL)
    assert answer.kind is Kind.COMPRESSED
    assert answer.tt is not None
    assert max(answer.tt.ranks) <= 2

    checked = verify_mps_circuit(answer.tt)
    assert checked["fidelity"] >= PHASE5
    assert checked["ancilla_disentangle"] >= PHASE5
    assert checked["ancilla_qubits"] == 1
    assert checked["max_bond"] == 2

    weights, _info = _prepared_probabilities(answer.tt)
    leaked = float(weights.sum() - weights[BRANCHES[0]] - weights[BRANCHES[1]])
    assert leaked <= 1.0 - PHASE5
    assert weights[BRANCHES[0]] > 0.0
    assert weights[BRANCHES[1]] > 0.0

    drawn = sample_from_tt(answer.tt, n_samples=64, seed=0)
    allowed = {tuple(bits) for bits in ( (0, 0, 0, 0, 0, 0), (0, 0, 1, 1, 0, 0) )}
    seen = {tuple(int(bit) for bit in row) for row in drawn}
    assert seen <= allowed
    assert seen == allowed
