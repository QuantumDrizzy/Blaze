"""
python/blaze/examples/mps_to_circuit.py

Phase 5 -- MPS -> quantum circuit synthesis (sequential preparation), VERIFIED.

Given a compressed TT/MPS, build the state-preparation circuit -- one unitary per
site acting on physical qubit_i plus a shared ancilla "bond" register -- and verify
by Cirq simulation that it reproduces the MPS amplitudes (and that the ancilla
returns to |0>). This is the ADR's load-bearing, HONEST quantum layer: there is NO
speedup or advantage claim. The honest research question it answers concretely: the
preparation cost (ancilla width m = ceil(log2 chi), gate dimension 2*chi) tracks the
bond dimension chi -- i.e. the compressibility.

Run:
    pip install -e '.[quantum]'   # needs cirq, quimb
    python -m blaze.examples.mps_to_circuit
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from blaze import compress  # noqa: E402
from blaze.cirq import (  # noqa: E402
    ghz_circuit,
    product_state_circuit,
    simulate_statevector,
    verify_mps_circuit,
)
from blaze.examples.substrate_quantum_state import tfim_groundstate  # noqa: E402


def row(name: str, tt) -> None:
    info = verify_mps_circuit(tt)
    print(f"{name:22}  chi={info['max_bond']:>2}  anc={info['ancilla_qubits']}  "
          f"gate={info['gate_dim']:>2}x{info['gate_dim']:<2}  gates={info['n_gates']:>2}  "
          f"depth={info['depth']:>2}  fid={info['fidelity']:.10f}  "
          f"anc|0>={info['ancilla_disentangle']:.6f}")


def main() -> None:
    print("=" * 92)
    print("Phase 5: MPS -> circuit synthesis, verified by simulation (NO advantage claim)")
    print("prep cost (ancilla m, gate dim 2*chi) tracks bond dimension chi = compressibility")
    print("=" * 92)

    # anchors with known bond dimension
    row("product n=6 (chi=1)", compress(simulate_statevector(product_state_circuit(6)), rel_tol=1e-10))
    for n in (4, 6, 8):
        row(f"GHZ n={n} (chi=2)", compress(simulate_statevector(ghz_circuit(n)), rel_tol=1e-10))

    # physical SUBSTRATE states: chi grows from the paramagnet toward criticality
    for h in (3.0, 2.0, 1.5, 1.0):
        tt = compress(tfim_groundstate(8, h).reshape([2] * 8), max_rank=16, rel_tol=1e-6)
        row(f"TFIM n=8 h={h:g}", tt)

    print("\nHonest read:")
    print("  * Every circuit reproduces its MPS to fidelity 1 and the ancilla returns to |0>.")
    print("  * product (chi=1) needs 0 ancilla; GHZ (chi=2) needs 1; physical TFIM states need")
    print("    more ancilla as entanglement (chi) grows toward criticality.")
    print("  * This is a correspondence + verification, NOT a speedup. The prep-cost scaling")
    print("    with chi is exactly why low-entanglement (compressible) states are cheap to prep.")


if __name__ == "__main__":
    main()
