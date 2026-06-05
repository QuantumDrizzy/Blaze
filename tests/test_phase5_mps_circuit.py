"""Phase 5 gate: MPS -> circuit synthesis reproduces the state by simulation.

Verifies the sequential-preparation construction: the synthesized circuit's
statevector matches the MPS amplitudes (fidelity ~1) and the ancilla bond register
returns to |0>. Anchored on states with known bond dimension. Skips if cirq is
absent (optional quantum dependency).
"""

import pytest

pytest.importorskip("cirq")

from blaze import compress  # noqa: E402
from blaze.cirq import (  # noqa: E402
    ghz_circuit,
    product_state_circuit,
    random_shallow_circuit,
    simulate_statevector,
    verify_mps_circuit,
)


def _run(circuit):
    tt = compress(simulate_statevector(circuit), rel_tol=1e-10)
    return verify_mps_circuit(tt)


def test_product_state_prep():
    info = _run(product_state_circuit(6))
    assert info["fidelity"] > 1 - 1e-9
    assert info["ancilla_qubits"] == 0          # chi=1 -> no ancilla
    assert info["ancilla_disentangle"] > 1 - 1e-9


def test_ghz_prep():
    for n in (3, 5, 7):
        info = _run(ghz_circuit(n))
        assert info["fidelity"] > 1 - 1e-9, f"GHZ n={n} fid={info['fidelity']}"
        assert info["max_bond"] == 2            # GHZ bond dimension is exactly 2
        assert info["ancilla_qubits"] == 1


def test_entangled_state_prep():
    info = _run(random_shallow_circuit(8, depth=3, seed=7))
    assert info["fidelity"] > 1 - 1e-8
    assert info["max_bond"] >= 3                # genuinely entangled (needs >1 ancilla)
    assert info["ancilla_disentangle"] > 1 - 1e-6
