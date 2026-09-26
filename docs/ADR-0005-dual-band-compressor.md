# ADR-0005: Blaze as a dual-band tensor compressor

**Status:** Proposed. A note to design from on 2026-10-05; nothing here is built or claimed.
**Date:** 2026-09-27
**Deciders:** Antonio
**Depends on:** ADR-0001 (honest scope), ADR-0002 (overlap and quantization), ADR-0004 (Blaze
speaks each engine's language)

## Context

QuBLAR became one thing on 2026-09-26: **an Ising photonic engine**. It has one contract
(probe → forward model → inference, QuBLAR ADR-008) and one kind of answer, a tri-state map
with a measured certainty. It is also held to one standard: ground truth, a control, an
evidence budget, and known limits filed next to results.

Blaze should get the same treatment: one identity, one contract, one kind of answer. The
pieces already exist and are measured:

| Band | What Blaze already does | Where |
|---|---|---|
| classical | TT-SVD compression, rank set per cut by the data's own spectrum; `.blz` on disk; int8 and 4-bit cores with the error composed | Phases 1–4, 6, 8 |
| quantum | an MPS **is** a state-preparation circuit: sequential synthesis (Schön 2005) reproduces the MPS at fidelity 1.0 | Phase 5 |
| both | inner products, fidelities and indexing without decompressing, exact to 1e-15, O(nχ³); runs at n = 40, where the dense form (17.6 TB) cannot exist | Phase 7 |
| neither | the diagnostic refuses what has no structure: Haar-random 0.38×, TT larger than dense | Phase 1, SUBSTRATE validation |

Blaze is unusual in one respect: **the same compressed object is classical data and a quantum
circuit at once.** A TT of a classical tensor and an MPS of a quantum state are the same
cores. That is the thing to name.

## Proposal

### 1. The identity (public)

> **Blaze: a dual-band tensor compressor.** Classical tensors and quantum states, one
> compressed form. It compresses what has structure, certifies what it kept, and declines
> the rest.

Alternatives, if "dual-band" does not land:
- *an entanglement compressor*: accurate, because the TT rank is the entanglement across each
  cut, but it sounds quantum-only;
- *a structure compressor*: honest, but generic.

"Adaptive" stays an internal word, not a public one. Internally it means that the rank at
every cut is chosen by the data's own singular spectrum under an error budget, and that the
band (bits or qubits) is chosen by the consumer.

### 2. The answer (the QuBLAR-grade part)

Every call returns a **verdict with a certificate**, the analogue of QuBLAR's tri-state map:

| Verdict | Meaning | Evidence carried |
|---|---|---|
| **compressed** | fits the budget | the TT-SVD truncation bound (‖A − Ã‖_F ≤ √Σ δ_k², Oseledets 2011), the measured error, and the composed quantization error |
| **declined** | TT would be larger than dense, or the spectrum is flat | the parameter count against dense, and the spectrum; the Haar control is the standing example |
| **undecided** | the rank hits its cap before the budget is met | the error reached at the cap, with the cut that bound it |

For the quantum band, "compressed" also carries the circuit (depth, two-qubit gate count) and
its fidelity against the MPS, as Phase 5 already measures.

### 3. The contract (like QuBLAR ADR-008)

```
  INGEST                    DECOMPOSE                   EMIT
  tensor | state vector  →  TT-SVD, rank by spectrum  →  classical band: cores, .blz, quantized
  (from any engine)         error budget declared        quantum band: state-prep circuit
                            verdict + certificate        both: compressed-space operations
```

Engines enter at INGEST through ADR-0004's bridges; nothing adopts Blaze's internals.

### 4. The trio, asynchronous and bare-metal (open question)

LYTH, QuBLAR and Blaze as stages that run concurrently on one machine, not as one program:
- QuBLAR produces posteriors and branch ensembles;
- Blaze compresses them and answers queries in compressed form;
- LYTH is the kernel language for the hot paths of both (the annealer sweep, TT contraction,
  and the quantized overlap), compiled to PTX and to MTLB assembly.

A trio like that needs two things decided, before any code:
1. **a shared, declared buffer layout** for TT cores and posteriors, so stages hand off without
   copies. LYTH's declared views (LYTH ADR-0028) are the natural way to say it.
2. **a hand-off protocol**: files today, which already work for QuBLAR → Blaze; shared memory
   with a queue later.

## Where Blaze sits: motor, compiler, compressor, QGPU

Blaze is the **compressor** in a four-piece stack: QuBLAR (motor), LYTH (compiler), Blaze
(compressor), MTLB (QGPU). The stack is defined in MTLB ADR-0002. Two facts tie Blaze to it:
- Blaze's int8 cores (Phase 8) are the operand format of MTLB's `ZIPPER2` instruction.
- Blaze's verdict is what lets QuBLAR fall back instead of failing: QuBLAR uses Blaze output
  only on *compressed* within its tolerance.

"Adaptive", as a working agreement: an instruction like "use Blaze here" needs no further
explanation. Read the goal, run INGEST → DECOMPOSE → EMIT, return the verdict.

## Not claimed

- No new compression ratio. Every number above is already measured in this repository.
- "Dual-band" is not a quantum advantage. The quantum band is state-preparation synthesis,
  verified in simulation, not run on hardware.
- The trio is a design question. Nothing here builds it.

## To decide on 2026-10-05

1. The name: dual-band, or one of the alternatives.
2. The verdict API: the fields, and where the certificate lives (`.blz` header, return value,
   or both).
3. The first gate that makes the identity real: a single call that returns *compressed*
   with a bound on a TFIM state, *declined* on Haar, and *undecided* at a forced rank cap, all
   in one test.
4. Whether the trio starts with files (cheap, and it already works) or with shared memory.
