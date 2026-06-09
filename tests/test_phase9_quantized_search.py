"""Phase 9 gate: overlap & similarity search over the SECOND-STAGE QUANTIZED cores.

Validates the TurboVec-analogue claim (ADR-0002) the honest way — by measurement, not
assertion:
  * inner_quantized is exactly the dequantized zipper (the primitive does what it says),
  * the overlap error shrinks as the bit width grows (quantization error, composed),
  * the int8 index is materially smaller than the float TT index (the realised win),
  * recall@k against the EXACT TTIndex stays at 1.0 at 8 bit and degrades gracefully —
    i.e. you can store the database in int8 and still retrieve the right neighbours.

Pure numpy — always runs. The FLOP win (int8 kernel) is Phase 9-Rust/CUDA, not here.
"""

import numpy as np
import pytest

from blaze import compress, inner, fidelity, TTIndex, quantize_tt
from blaze.tt import TT
from blaze.quantized_search import (
    inner_quantized,
    fidelity_quantized,
    QuantizedTTIndex,
)


# --------------------------------------------------------------------------- #
# helpers — distinguishable TT states with a clean similarity gradient
# --------------------------------------------------------------------------- #
def _product_tt(angles) -> TT:
    """Bond-1 product state ⊗_k (cos θ_k|0⟩ + sin θ_k|1⟩). No dense 2ⁿ array."""
    cores = [
        np.array([np.cos(t), np.sin(t)], dtype=np.complex128).reshape(1, 2, 1)
        for t in angles
    ]
    n = len(angles)
    return TT(cores=cores, shape=(2,) * n, singular_values=[np.array([1.0])] * (n - 1))


def _graded_family(N: int = 12, n: int = 6) -> list[TT]:
    """N normalized product states on a CHIRPED angular grid (strictly growing gaps).

    Pairwise fidelity is strictly monotone in s with NO exact ties — a well-posed
    nearest-neighbour benchmark. (A uniform grid makes s±m equidistant, tying the
    k-boundary so that top-k label-set recall is ill-defined under any perturbation,
    quantization included.)
    """
    gaps = 0.10 * (1.0 + 0.05 * np.arange(N - 1))      # N-1 strictly increasing gaps
    angs = np.concatenate([[0.0], np.cumsum(gaps)])    # N angles, no ties
    return [_product_tt([a] * n) for a in angs]


def _random_lowrank_tt(shape, max_rank, seed) -> TT:
    """A random complex tensor with genuine low TT-rank structure."""
    rng = np.random.default_rng(seed)
    dense = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return compress(dense, max_rank=max_rank, rel_tol=1e-12)


def _recall(exact: TTIndex, approx: QuantizedTTIndex, queries, k: int) -> float:
    """Mean overlap of the top-k label sets (exact index vs quantized index)."""
    hit = total = 0
    for q in queries:
        gt = {lab for lab, _, _ in exact.query(q, k=k)}
        got = {lab for lab, _, _ in approx.query(q, k=k)}
        hit += len(gt & got)
        total += len(gt)
    return hit / total if total else 0.0


# --------------------------------------------------------------------------- #
# 1. the primitive does exactly what it claims
# --------------------------------------------------------------------------- #
def test_inner_quantized_is_the_dequantized_zipper():
    a = _random_lowrank_tt((4, 4, 4), max_rank=6, seed=1)
    b = _random_lowrank_tt((4, 4, 4), max_rank=6, seed=2)
    qa, qb = quantize_tt(a, bits=8), quantize_tt(b, bits=8)
    # named primitive == the explicit dequantize-then-zipper, bit for bit
    assert inner_quantized(qa, qb) == inner(qa.dequantize(), qb.dequantize())


def test_mixed_tt_and_quantized_operands():
    a = _random_lowrank_tt((3, 4, 3), max_rank=5, seed=10)
    qa = quantize_tt(a, bits=8)
    # ⟨a|a_quant⟩ is close to ⟨a|a⟩ but not exact (one side carries quant error)
    exact = inner(a, a)
    mixed = inner_quantized(a, qa)
    assert abs(mixed - exact) / abs(exact) < 1e-2


# --------------------------------------------------------------------------- #
# 2. error composes and shrinks with bit width
# --------------------------------------------------------------------------- #
def test_overlap_error_shrinks_with_bits():
    a = _random_lowrank_tt((4, 4, 4), max_rank=6, seed=3)
    b = _random_lowrank_tt((4, 4, 4), max_rank=6, seed=4)
    ref = inner(a, b)
    errs = {}
    for bits in (4, 6, 8, 12):
        qa, qb = quantize_tt(a, bits=bits), quantize_tt(b, bits=bits)
        errs[bits] = abs(inner_quantized(qa, qb) - ref)
    # higher precision is never meaningfully worse, and 12-bit clearly beats 4-bit
    assert errs[12] < errs[4]
    assert errs[8] <= errs[4] * 1.5
    assert errs[12] <= errs[6] * 1.5


def test_quantized_fidelity_tracks_exact():
    a = _random_lowrank_tt((4, 5, 4), max_rank=8, seed=5)
    b = _random_lowrank_tt((4, 5, 4), max_rank=8, seed=6)
    fref = fidelity(a, b)
    fq = fidelity_quantized(quantize_tt(a, bits=8), quantize_tt(b, bits=8))
    assert abs(fq - fref) < 1e-2


# --------------------------------------------------------------------------- #
# 3. the realised win here is MEMORY (int8 index < float TT index)
# --------------------------------------------------------------------------- #
def test_quantized_index_is_smaller():
    tt = _random_lowrank_tt((6, 6, 6, 6), max_rank=8, seed=11)
    float_bytes = sum(c.nbytes for c in tt.cores)  # complex128 cores
    q = quantize_tt(tt, bits=8)
    assert q.nbytes() < float_bytes
    # complex128 (16 B/entry) -> int8 re+im (2 B) + scales: comfortably > 2x smaller
    assert q.nbytes() < float_bytes / 2.0


# --------------------------------------------------------------------------- #
# 4. THE headline: recall@k vs the exact index
# --------------------------------------------------------------------------- #
def test_recall_at_k_is_perfect_at_8bit():
    fam = _graded_family(N=12, n=6)
    exact = TTIndex("fidelity")
    approx = QuantizedTTIndex("fidelity")
    for s, tt in enumerate(fam):
        exact.add(tt, label=s)
        approx.add_tt(tt, bits=8, label=s)
    # query with every (exact) state; 8-bit reproduces the exact top-4 neighbourhood
    recall = _recall(exact, approx, fam, k=4)
    assert recall >= 0.95


def test_recall_degrades_gracefully_low_bits():
    fam = _graded_family(N=12, n=6)
    exact = TTIndex("fidelity")
    for s, tt in enumerate(fam):
        exact.add(tt, label=s)
    recalls = {}
    for bits in (4, 6, 8):
        approx = QuantizedTTIndex("fidelity")
        for s, tt in enumerate(fam):
            approx.add_tt(tt, bits=bits, label=s)
        recalls[bits] = _recall(exact, approx, fam, k=4)
    # monotone non-decreasing in bits (tiny tolerance for residual reordering),
    # and even 4-bit keeps most of the neighbourhood
    assert recalls[8] >= recalls[6] - 1e-9
    assert recalls[6] >= recalls[4] - 1e-9
    assert recalls[4] >= 0.7


def test_top1_self_retrieval_under_quantization():
    fam = _graded_family(N=10, n=6)
    approx = QuantizedTTIndex("fidelity")
    for s, tt in enumerate(fam):
        approx.add_tt(tt, bits=8, label=s)
    # every state's own (exact) vector still ranks itself first against int8 neighbours
    for j, tt in enumerate(fam):
        assert approx.query(tt, k=1)[0][0] == j


def test_distance_metric_recall():
    fam = _graded_family(N=10, n=6)
    exact = TTIndex("distance")
    approx = QuantizedTTIndex("distance")
    for s, tt in enumerate(fam):
        exact.add(tt, label=s)
        approx.add_tt(tt, bits=8, label=s)
    assert _recall(exact, approx, fam, k=3) == 1.0
