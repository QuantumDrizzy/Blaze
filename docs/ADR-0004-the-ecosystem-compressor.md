# ADR-0004: Blaze speaks each engine's language

**Status:** Accepted (direction; each bridge gets its own change with its own gate)
**Date:** 2026-09-26
**Deciders:** Antonio

## Context

Blaze is a specialised compressor: TT/MPS, quantization, and operations in the compressed
form. On 2026-09-26 the ecosystem took its shape:

- QuBLAR is an Ising photonic engine, with SUBSTRATE as its lab;
- DRiFT is an Ising engine, with computronium as its field;
- Unibit is a processor programme (CPU → TPU → QPU);
- LYTH compiles and validates Unibit.

Blaze is none of these. It moves their data, compressed. **With no data it does nothing**:
if QuBLAR produces no ghost bits, Blaze has nothing to compress. That is correct behaviour,
not a gap.

## Decision

Blaze keeps one core (TT, quantization, compressed-space operations) and speaks each
neighbour's format at the boundary. No neighbour adopts Blaze's internals.

| Link | What crosses it | Status |
|---|---|---|
| **QuBLAR → Blaze** | exact region-of-interest posteriors over ghost bits (2ⁿ tensors, one mode per bit), and branch ensembles | **working**: QuBLAR's `tools/branches_blaze.py` (TT rank 1–2, marginals to 7 × 10⁻⁷) |
| **DRiFT → Blaze** | Ising ground and low-lying states, and MPS from DRiFT's tensor-network solvers | open; format-compatible (both are MPS) |
| **LYTH ↔ Blaze** | Blaze's hot kernels (TT contraction, the quantized overlap of Phase 9-real) written as `.lyth`: movement declared, intensity derived and checked | open; Phase 9-real is the first candidate |
| **LYTH ↔ Blaze ↔ Unibit** | the same `.lyth` kernel compiled to PTX (GPU) and to Unibit assembly (`lyth-uasm`), with TT cores as operands of Unibit's tensor-network unit; results checked against Blaze's Python reference | open, after LYTH ↔ Blaze |
| **VENTUS** | nothing: its data are small and exact | not needed |

**Rule.** A bridge is real when a gate in the receiving repository reproduces Blaze's Python
reference to a stated tolerance, the same discipline as Phases 2–9.

## Consequences

- Blaze stays the specialised compressor, **not a general one**, as ADR-0001's honest scope
  says. What grows is the number of formats it reads and writes, not its claims.
- The chain LYTH ↔ DRiFT ↔ QuBLAR ↔ Blaze closes one bridge at a time, each with its gate.
