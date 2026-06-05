# Phase 5 — MPS → circuit synthesis (sequential preparation), verified

**Date:** 2026-06-05 · cirq + Python. **No advantage / speedup claim** (ADR-0001 §Cirq).

## What Phase 5 is

The ADR's load-bearing, honest quantum layer: given a TT/MPS produced by the
compressor, **construct the state-preparation circuit and verify — by simulation,
on small n — that it reproduces the MPS amplitudes.** The deliverable is the
*correspondence + verification*, plus the honest research question: does the
preparation cost track the compressibility?

## Construction (sequential preparation, Schön et al. 2005)

1. **Right-canonicalize** the MPS (LQ sweep) so each core, as a map from the left
   bond to (physical ⊗ right bond), is an isometry; the global norm is folded into
   and normalized out of the first core.
2. For each site build a unitary `U_i` on **(physical qubit_i ⊗ ancilla bond
   register)** of `m = ⌈log₂χ⌉` ancilla qubits, by completing the core isometry to a
   full unitary (null-space completion).
3. Apply `U_1 … U_N` **left→right** on a shared ancilla register. The ancilla starts
   in `|0…0⟩` and (because `r_N = 1`) returns to `|0…0⟩`; the physical register is
   left holding `|ψ⟩`.

## Gate — verified by simulation (fidelity 1, ancilla disentangles)

| state | bond χ | ancilla m | gate dim | n_gates | fidelity | ancilla→\|0⟩ |
|-------|------:|----------:|---------:|--------:|---------:|-------------:|
| product n=6 | 1 | 0 | 2×2 | 6 | 1.0000000000 | 1.000000 |
| GHZ n=4 | 2 | 1 | 4×4 | 4 | 1.0000000000 | 1.000000 |
| GHZ n=6 | 2 | 1 | 4×4 | 6 | 1.0000000000 | 1.000000 |
| GHZ n=8 | 2 | 1 | 4×4 | 8 | 1.0000000000 | 1.000000 |
| TFIM n=8 h=3 (paramagnet) | 5 | 3 | 16×16 | 8 | 1.0000000000 | 1.000000 |
| TFIM n=8 h=2 | 6 | 3 | 16×16 | 8 | 1.0000000000 | 1.000000 |
| TFIM n=8 h=1.5 | 8 | 3 | 16×16 | 8 | 1.0000000000 | 1.000000 |
| TFIM n=8 h=1 (critical) | 10 | 4 | 32×32 | 8 | 1.0000000000 | 1.000000 |

Every synthesized circuit reproduces its MPS to fidelity 1, and the ancilla returns
to `|0⟩` (so the physical register is left in a pure `|ψ⟩`, unentangled from the
ancilla). Verified on golden anchors (product χ=1, GHZ χ=2) and on **physical
SUBSTRATE ground states**.

## The honest research finding

**Preparation cost tracks the bond dimension χ = entanglement = compressibility.**
Product (χ=1) needs 0 ancilla and single-qubit gates; GHZ (χ=2) needs 1 ancilla and
4×4 gates; the physical TFIM ground states need wider ancilla as entanglement grows
toward criticality (χ 5→10, ancilla 3→4, gate 16×16→32×32). This is *exactly why
low-entanglement (compressible) states are cheap to prepare* — and it is a
**correspondence, not a speedup**. There is no advantage claim (ADR §Cirq).

## Scope and honest limits

- Verification is **exact statevector simulation on small n** (≤ ~12 qubits incl.
  ancilla here). It proves the construction is correct, not that it scales cheaply.
- The MPS may itself approximate the true state (Phase 4). This gate measures the
  **circuit ↔ MPS** fidelity (the synthesis); the **MPS ↔ true-state** truncation
  error is the separate Phase 4 quantity. `[KNOWN_LIMIT]`
- Ancilla width and gate dimension grow as χ = 2ᵐ; large-χ (volume-law) states need
  wide gates — expected, and precisely the point of the cost↔χ correspondence.

## Reproduce

```bash
pip install -e '.[quantum]'        # cirq, quimb
python -m blaze.examples.mps_to_circuit
pytest tests/test_phase5_mps_circuit.py -v
```

## Environment note

`cirq-core` pulls in `pandas`; `pandas < 2.2.2` is ABI-incompatible with `numpy ≥ 2`
(`ValueError: numpy.dtype size changed`). Pin **`pandas >= 2.2.2, < 3`** (the 2.x
line is numpy-2 compatible and avoids the pandas-3 API breaks).
