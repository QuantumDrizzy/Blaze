"""
python/blaze/examples/quantum_compression.py

Cirq integration demo + golden tests for Fase 1.

This proves the *implementation is correct* using physics guarantees:
- GHZ: exact bond dimension 2 (independent of n)
- Product state: bond dimension 1
- Shallow random: modest rank growth

These are NOT the test of whether Blaze is a useful general compressor (see classical_benchmark.py).

Run:
    pip install -e '.[quantum]'
    python -m blaze.examples.quantum_compression
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from blaze import compress  # noqa: E402
from blaze.cirq import (  # noqa: E402
    compress_circuit_state,
    fidelity,
    sample_from_tt,
    ghz_circuit,
    product_state_circuit,
    random_shallow_circuit,
    simulate_statevector,
)


def run_golden(name: str, circuit, max_rank: int | None = None, rel_tol: float = 1e-6):
    print(f"\n=== {name} ===")
    tt, info = compress_circuit_state(circuit, max_rank=max_rank, rel_tol=rel_tol, verbose=False)
    exact = simulate_statevector(circuit)
    fid = fidelity(tt, exact)
    print(f"n_qubits     : {info['n_qubits']}")
    print(f"TT ranks     : {tt.ranks}")
    print(f"eff ranks 1e-6: {info['effective_ranks_1e-6']}")
    print(f"nparams TT   : {info['nparams']} / original {info['original_params']}")
    print(f"fidelity     : {fid:.10f}")
    # quick sampling sanity (only for d=2)
    if all(d == 2 for d in tt.shape):
        samples = sample_from_tt(tt, n_samples=256, seed=42)
        # For GHZ we should see only all-0 and all-1
        if "GHZ" in name:
            unique = {tuple(s) for s in samples}
            print(f"GHZ samples unique patterns (should be ~2): {len(unique)}")
    return fid


def main():
    print("Blaze + Cirq — Quantum State Compression (golden verification layer)")
    print("These cases have *known* low bond dimension by construction.\n")

    # GHZ — the star
    for n in [6, 10, 14]:
        c = ghz_circuit(n)
        run_golden(f"GHZ n={n} (exact bond dim 2)", c, max_rank=4, rel_tol=1e-8)

    # Product state — should be rank 1 everywhere
    c = product_state_circuit(8)
    run_golden("Product state (zero entanglement)", c, max_rank=2, rel_tol=1e-6)

    # Shallow random — entanglement grows but slowly
    c = random_shallow_circuit(10, depth=4, seed=123)
    run_golden("Shallow random circuit (depth 4)", c, max_rank=16, rel_tol=1e-4)

    print("\n=== Summary ===")
    print("If GHZ gives fidelity > 1-1e-10 with rank<=2 and product gives rank=1, the TT-SVD core is correct.")
    print("These are cheap to run and act as regression gates before any Rust/CUDA work.")
    print("The real (risky) test of usefulness is in classical_benchmark.py.")


if __name__ == "__main__":
    main()
