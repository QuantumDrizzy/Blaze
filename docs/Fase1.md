# Fase 1 — Pure Python Prototype (executable spec + honest classical gate)

Status: complete (this directory)

## What was built
- Full `blaze/` package with stable public API (`compress`, `TT`, `analyze_compressibility`).
- TT-SVD implementation with correct truncation, reconstruct, error, ratio, effective ranks.
- Diagnostics that turn the architectural `[KNOWN_LIMIT]` into numbers (singular decay per unfolding).
- `blaze.cirq`: real integration. GHZ etc. as golden tests that must pass for impl correctness.
- `examples/classical_benchmark.py`: the one that matters for claim (b). Multiple structured classical generators + matrix-SVD baseline + verdict per case.
- `examples/quantum_compression.py`: the physics verification layer (does not prove usefulness).
- CLI stubs + pyproject with optional `[quantum]`.
- dtype policy respected: f32/f64 + c128 in core; c64 deferred.
- No custom binary format (Fase 2); cores are numpy arrays (easy .npy export if wanted).
- No Rust/CUDA yet — by design.

## How to run the gates (Antonio review)
```powershell
cd C:\Users\Drizzy\desktop\Blaze
python -m pip install -e ".[quantum]"   # or without [quantum] for bench only

python -m blaze.examples.classical_benchmark
python -m blaze.examples.quantum_compression
```

Or via the entry point:
```powershell
blaze --bench
blaze --quantum
```

## What the classical benchmark must show (pass criteria for Fase 1)
1. The "low-rank TT ground truth + noise" case recovers with very low error at low max_rank (sanity + impl correct).
2. The smooth field and hyperspectral-like cases report their effective ranks + decay summary. If median effective rank stays << dimension of unfoldings, the data class has hope.
3. Direct comparison vs single-unfolding matrix SVD (params + achieved error). TT should be competitive or better on the structured cases; on random it should not win (expected).
4. All numbers reproducible, no hidden "easy" data.

If (2)+(3) look promising on the hyperspectral-like and field cases, we have a signal worth taking to Rust + CUDA. If not, we narrow scope or improve (mode ordering, better algos) before investing.

## Next (after review of these numbers)
- Fix any bugs found in review.
- Write ADR / update architecture.md with the measured data from Fase 1.
- Then Fase 2: Rust core (faer CPU ref) + I/O + CLI, still no CUDA.
- Only after classical claim (b) survives Fase 1+2 do we touch kernels.

## Known limits in this prototype
- TT-SVD materializes the full tensor (as designed).
- Reconstruction for large tensors will OOM (same as planned).
- No mode reordering heuristic yet (user responsibility + documented).
- SVD is scipy (CPU). CUDA version will use cuSolver via cudarc.
- Sampling MPS is basic (correctness over speed).

See root README and the original architecture proposal for the full contract.
