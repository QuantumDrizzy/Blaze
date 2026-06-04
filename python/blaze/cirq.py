"""
blaze.cirq

Serious Cirq integration for Fase 1.

Purpose:
- Quantum states (statevectors) are the *canonical* example of high-dimensional
  tensors that admit efficient MPS/TT representations when entanglement is low.
- This module turns Cirq circuits into verifiable compression tasks.
- Golden cases (GHZ = exact bond dimension 2; product states = bond 1) prove
  that the TT-SVD implementation is *correct*, not just lucky on easy data.
- Real claim (b) for classical data is tested elsewhere (classical_benchmark).

Cirq is an optional dependency (pip install -e .[quantum]).
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import cirq  # type: ignore
except ImportError:  # pragma: no cover
    cirq = None  # type: ignore

from .tt import TT, compress


def _require_cirq():
    if cirq is None:
        raise ImportError(
            "blaze.cirq requires cirq. Install with: pip install -e '.[quantum]' "
            "or 'pip install cirq-core' (for basic simulator support)."
        )


def simulate_statevector(circuit: Any, qubit_order: Any = None) -> np.ndarray:
    """Return the final state vector as complex128 ndarray of shape (2,)*n."""
    _require_cirq()
    sim = cirq.Simulator(dtype=np.complex128)
    if qubit_order is None:
        # default order: sorted by cirq line order or insertion
        qubit_order = sorted(circuit.all_qubits())
    result = sim.simulate(circuit, qubit_order=qubit_order)
    psi = result.final_state_vector
    n = int(np.round(np.log2(psi.size)))
    assert 2**n == psi.size, "statevector size not power of 2"
    return psi.reshape((2,) * n)


def compress_circuit_state(
    circuit: Any,
    max_rank: int | None = None,
    rel_tol: float = 1e-6,
    verbose: bool = False,
    qubit_order: Any = None,
) -> tuple[TT, dict]:
    """Compress the statevector of a Cirq circuit to TT/MPS.

    Returns:
        (TT, info_dict with 'n_qubits', 'exact_norm', 'effective_ranks', ...)
    """
    _require_cirq()
    psi = simulate_statevector(circuit, qubit_order=qubit_order)
    n = psi.ndim
    tt = compress(psi, max_rank=max_rank, rel_tol=rel_tol, verbose=verbose)
    info = {
        "n_qubits": n,
        "shape": psi.shape,
        "exact_norm": float(np.linalg.norm(psi)),
        "tt_ranks": tt.ranks,
        "effective_ranks_1e-6": tt.effective_ranks(1e-6),
        "effective_ranks_1e-4": tt.effective_ranks(1e-4),
        "nparams": tt.nparams(),
        "original_params": int(psi.size),
    }
    return tt, info


def fidelity(tt: TT, exact_state: np.ndarray) -> float:
    """Compute |<psi_exact | psi_tt>|^2 (state fidelity).

    Both are assumed normalized or we normalize internally.
    Works for the (2,2,...,2) reshaping used by Cirq.
    """
    approx = tt.reconstruct().astype(np.complex128, copy=False)
    exact = exact_state.astype(np.complex128, copy=False).ravel()
    approx = approx.ravel()
    # normalize both
    exact /= np.linalg.norm(exact) + 1e-30
    approx /= np.linalg.norm(approx) + 1e-30
    inner = np.vdot(exact, approx)  # <exact|approx>
    return float(np.abs(inner) ** 2)


def sample_from_tt(tt: TT, n_samples: int = 1024, seed: int | None = None) -> np.ndarray:
    """Sample bitstrings from the MPS/TT (assuming physical dims are 2).

    This is the efficient MPS sampling algorithm (left-to-right conditional).
    Returns int array of shape (n_samples, n_qubits) with 0/1 entries.
    """
    rng = np.random.default_rng(seed)
    cores = tt.cores
    n = len(cores)
    assert all(c.shape[1] == 2 for c in cores), "sample_from_tt only for qubit (d=2) tensors"

    samples = np.zeros((n_samples, n), dtype=np.int8)

    for s in range(n_samples):
        # current bond vector (distribution amplitude on the bond)
        # start from left bond = 1 (scalar 1.0)
        bond_vec = np.array([1.0 + 0j])  # shape (r_left,)

        bits = []
        for k in range(n):
            core = cores[k]  # (r_l, 2, r_r)
            # contract current bond
            # temp: (2, r_r) amplitudes for |0> and |1> on this qubit
            temp = np.tensordot(bond_vec, core, axes=([0], [0]))  # (2, r_r)
            # marginal prob for 0 vs 1: sum over right bond |amp|^2
            p0 = np.sum(np.abs(temp[0]) ** 2).real
            p1 = np.sum(np.abs(temp[1]) ** 2).real
            p0 = max(0.0, min(1.0, p0 / (p0 + p1 + 1e-30)))
            bit = 0 if rng.random() < p0 else 1
            bits.append(bit)
            # update bond_vec to the chosen slice, renormalized
            chosen = temp[bit]
            norm = np.linalg.norm(chosen) + 1e-30
            bond_vec = (chosen / norm).astype(np.complex128)

        samples[s] = bits
    return samples


def ghz_circuit(n_qubits: int) -> Any:
    """Exact GHZ state: |00...0> + |11...1> / sqrt(2). Bond dimension exactly 2."""
    _require_cirq()
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit()
    circuit.append(cirq.H(qubits[0]))
    for i in range(n_qubits - 1):
        circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
    return circuit


def product_state_circuit(n_qubits: int, angles: list[float] | None = None) -> Any:
    """Product state (zero entanglement). Should compress to bond dimension 1."""
    _require_cirq()
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit()
    if angles is None:
        angles = [0.3 * i for i in range(n_qubits)]
    for q, a in zip(qubits, angles):
        circuit.append(cirq.Rx(rads=a)(q))
    return circuit


def random_shallow_circuit(n_qubits: int, depth: int = 3, seed: int = 42) -> Any:
    """Shallow random circuit (modest entanglement). Good for seeing rank growth."""
    _require_cirq()
    rng = np.random.default_rng(seed)
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit()
    for d in range(depth):
        # single qubit
        for q in qubits:
            if rng.random() < 0.5:
                circuit.append(cirq.Rx(rads=rng.uniform(0, np.pi))(q))
            else:
                circuit.append(cirq.Rz(rads=rng.uniform(0, np.pi))(q))
        # entangling
        for i in range(n_qubits - 1):
            if rng.random() < 0.7:
                circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
    return circuit
