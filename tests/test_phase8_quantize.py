"""Phase 8 gate: second-stage core quantization, honest and composable.

Locks the ADR-0002 invariants:
  * round-trip error is MONOTONE in bit width (more bits → less error),
  * quantization only ADDS error (combined ≥ truncation-only) and the combined
    error converges to the truncation error as bits grow,
  * storage is the real bit-packed size and shrinks with fewer bits,
  * per-bond scaling beats per-core on cores carrying the singular values,
  * golden: a GHZ state survives int8 with fidelity ~1,
  * complex (quantum) and real tensors both round-trip.

Pure numpy — always runs. Fidelity is measured with the Phase 7 overlap (the two
phases compose).
"""

import numpy as np
import pytest

from blaze import compress, quantize_tt, fidelity
from blaze.tt import TT


def _ghz(n: int) -> np.ndarray:
    psi = np.zeros(2**n, dtype=np.complex128)
    psi[0] = psi[-1] = 1.0 / np.sqrt(2.0)
    return psi


def _rel_err(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm((a - b).ravel()) / np.linalg.norm(a.ravel()))


# --------------------------------------------------------------------------- #
# 1. monotone in bits — more bits, less error
# --------------------------------------------------------------------------- #
def test_error_monotone_in_bits_complex():
    # A random complex state (spread entries) so the quantization STEP — not the
    # float32 scale floor — sets the error. (GHZ's exact ±1/√2 entries quantize
    # losslessly and would sit at the float32 floor, a degenerate case.)
    rng = np.random.default_rng(11)
    dense = rng.standard_normal((6, 6, 6)) + 1j * rng.standard_normal((6, 6, 6))
    tt = compress(dense, max_rank=6, rel_tol=1e-12)   # full-rank → near-lossless TT
    ref = tt.reconstruct()                            # isolate pure quantization error
    errs = []
    for bits in (4, 6, 8, 12):
        q = quantize_tt(tt, bits=bits, granularity="per_bond")
        errs.append(_rel_err(ref, q.dequantize().reconstruct()))
    assert all(errs[i] > errs[i + 1] for i in range(len(errs) - 1)), errs


def test_error_monotone_in_bits_real():
    rng = np.random.default_rng(0)
    dense = rng.standard_normal((4, 5, 4))
    tt = compress(dense, max_rank=8, rel_tol=1e-10)
    ref = tt.reconstruct()
    e4 = _rel_err(ref, quantize_tt(tt, bits=4).dequantize().reconstruct())
    e8 = _rel_err(ref, quantize_tt(tt, bits=8).dequantize().reconstruct())
    assert e8 < e4


# --------------------------------------------------------------------------- #
# 2. error composes: quantization only adds, and vanishes as bits → ∞
# --------------------------------------------------------------------------- #
def test_quantization_only_adds_error():
    rng = np.random.default_rng(1)
    dense = rng.standard_normal((6, 6, 6)) + 1j * rng.standard_normal((6, 6, 6))
    tt = compress(dense, max_rank=5, rel_tol=1e-8)       # genuine truncation error
    trunc_err = tt.rel_error(dense)

    q8 = quantize_tt(tt, bits=8, granularity="per_bond")
    combined8 = _rel_err(dense, q8.dequantize().reconstruct())
    # combined ≥ truncation alone (quantizing cannot reduce error)
    assert combined8 >= trunc_err - 1e-12
    # and at high bit width it converges back toward the truncation floor
    q14 = quantize_tt(tt, bits=14, granularity="per_bond")
    combined14 = _rel_err(dense, q14.dequantize().reconstruct())
    assert combined14 <= combined8
    assert abs(combined14 - trunc_err) <= 5e-3 * max(trunc_err, 1e-6) + 1e-4


# --------------------------------------------------------------------------- #
# 3. storage shrinks with fewer bits, and nbytes is the real packed size
# --------------------------------------------------------------------------- #
def test_storage_shrinks_with_bits():
    tt = compress(_ghz(10).reshape((2,) * 10), rel_tol=1e-12)
    tt_bytes = sum(c.nbytes for c in tt.cores)
    b8 = quantize_tt(tt, bits=8).nbytes()
    b4 = quantize_tt(tt, bits=4).nbytes()
    assert b4 < b8 < tt_bytes  # fewer bits → less storage, both below the TT


def test_metadata_tradeoff_tiny_vs_large_cores():
    # GHZ has bond-2 (tiny) cores: per-bond scale metadata dominates, so per_core
    # stores SMALLER. This is the honest [KNOWN_LIMIT] — metadata erodes the ratio.
    ghz = compress(_ghz(10).reshape((2,) * 10), rel_tol=1e-12)
    assert (
        quantize_tt(ghz, bits=8, granularity="per_core").nbytes()
        < quantize_tt(ghz, bits=8, granularity="per_bond").nbytes()
    )

    # Larger cores: codes dominate, metadata is small, int8 approaches the 8x bound
    # (complex128 16 B/param → int8 re+im 2 B/param).
    rng = np.random.default_rng(5)
    big = rng.standard_normal((8, 8, 8, 8)) + 1j * rng.standard_normal((8, 8, 8, 8))
    tt = compress(big, max_rank=8, rel_tol=1e-10)
    ratio8 = quantize_tt(tt, bits=8, granularity="per_bond").compression_ratio_over_tt(tt)
    assert ratio8 > 6.5, ratio8


# --------------------------------------------------------------------------- #
# 4. per-bond beats per-core where it matters (the S·Vh core)
# --------------------------------------------------------------------------- #
def test_per_bond_beats_per_core():
    # a state with a spread singular spectrum so the last core has varied column norms
    rng = np.random.default_rng(2)
    dense = rng.standard_normal((8, 8, 8)) + 1j * rng.standard_normal((8, 8, 8))
    tt = compress(dense, max_rank=8, rel_tol=1e-10)
    ref = tt.reconstruct()
    e_core = _rel_err(ref, quantize_tt(tt, bits=6, granularity="per_core").dequantize().reconstruct())
    e_bond = _rel_err(ref, quantize_tt(tt, bits=6, granularity="per_bond").dequantize().reconstruct())
    assert e_bond <= e_core


# --------------------------------------------------------------------------- #
# 5. golden — GHZ survives int8 at fidelity ~1 (Phase 7 overlap measures it)
# --------------------------------------------------------------------------- #
def test_golden_ghz_int8_fidelity():
    tt = compress(_ghz(6).reshape((2,) * 6), rel_tol=1e-12)
    q = quantize_tt(tt, bits=8, granularity="per_bond")
    fid = fidelity(tt, q.dequantize())     # ⟨orig|dequant⟩ in compressed space
    assert fid > 1 - 1e-3, fid


def test_complex_and_real_roundtrip_shapes():
    # complex
    tt_c = compress(_ghz(5).reshape((2,) * 5), rel_tol=1e-12)
    dq_c = quantize_tt(tt_c, bits=8).dequantize()
    assert dq_c.cores[0].dtype == np.complex128
    assert dq_c.shape == tt_c.shape
    # real
    rng = np.random.default_rng(3)
    tt_r = compress(rng.standard_normal((3, 4, 3)), max_rank=4, rel_tol=1e-8)
    dq_r = quantize_tt(tt_r, bits=8).dequantize()
    assert dq_r.cores[0].dtype == np.float64
    assert dq_r.shape == tt_r.shape
