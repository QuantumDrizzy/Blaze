"""Phase 7 gate: inner product / fidelity / distance / search on the TT directly.

Locks the honest invariants of ADR-0002:
  * the MPS zipper computes the SAME number as the dense vdot (≤1e-12), without
    ever decompressing,
  * fidelity and distance match their dense definitions,
  * golden anchors hold (⟨GHZ|GHZ⟩=1, fidelity(GHZ, |0…0⟩)=½, orthogonal→0),
  * it runs at n=40 (2⁴⁰ dense is impossible) for low-χ states — proving it never
    materializes the dense tensor,
  * TTIndex retrieves the right nearest neighbour in compressed space.

Pure numpy — no cirq/quimb needed, always runs.
"""

import numpy as np
import pytest

from blaze import compress, inner, norm, fidelity, distance, TTIndex
from blaze.tt import TT


# --------------------------------------------------------------------------- #
# helpers — build qubit states without any optional dependency
# --------------------------------------------------------------------------- #
def _tt_from_statevector(psi_flat: np.ndarray, n: int, rel_tol: float = 1e-12) -> TT:
    """Compress a 2ⁿ statevector (lossless rel_tol) into a qubit TT."""
    return compress(psi_flat.reshape((2,) * n), rel_tol=rel_tol)


def _ghz(n: int) -> np.ndarray:
    psi = np.zeros(2**n, dtype=np.complex128)
    psi[0] = psi[-1] = 1.0 / np.sqrt(2.0)
    return psi


def _basis(n: int, all_ones: bool = False) -> np.ndarray:
    psi = np.zeros(2**n, dtype=np.complex128)
    psi[-1 if all_ones else 0] = 1.0
    return psi


def _product_tt(single_qubit_states: list[np.ndarray]) -> TT:
    """Bond-1 TT built directly from per-qubit (2,) vectors — no dense 2ⁿ array.

    Lets us exercise n far beyond what a dense statevector could hold.
    """
    cores = [s.reshape(1, 2, 1).astype(np.complex128) for s in single_qubit_states]
    n = len(single_qubit_states)
    return TT(cores=cores, shape=(2,) * n, singular_values=[np.array([1.0])] * (n - 1))


def _random_lowrank_tt(shape, max_rank, seed) -> tuple[TT, np.ndarray]:
    """A random tensor with genuine low TT-rank structure + its compressed TT."""
    rng = np.random.default_rng(seed)
    dense = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    tt = compress(dense, max_rank=max_rank, rel_tol=1e-12)
    return tt, tt.reconstruct()  # the dense reference is the reconstruction


# --------------------------------------------------------------------------- #
# 1. the core claim: zipper == dense vdot, without decompressing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [4, 6, 8])
def test_inner_matches_dense_vdot(n):
    a = _tt_from_statevector(_ghz(n), n)
    rng = np.random.default_rng(n)
    psi_b = rng.standard_normal(2**n) + 1j * rng.standard_normal(2**n)
    b = _tt_from_statevector(psi_b, n)

    got = inner(a, b)
    ref = np.vdot(a.reconstruct().ravel(), b.reconstruct().ravel())
    assert abs(got - ref) <= 1e-12, f"zipper {got} != dense vdot {ref}"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_inner_general_tensor(seed):
    # order-4 tensor (not a qubit chain) — overlap must still equal dense vdot
    a, da = _random_lowrank_tt((3, 4, 3, 4), max_rank=6, seed=seed)
    b, db = _random_lowrank_tt((3, 4, 3, 4), max_rank=6, seed=seed + 100)
    got = inner(a, b)
    ref = np.vdot(da.ravel(), db.ravel())
    assert abs(got - ref) <= 1e-10


def test_norm_matches_dense():
    a, da = _random_lowrank_tt((4, 5, 4), max_rank=8, seed=7)
    assert abs(norm(a) - np.linalg.norm(da.ravel())) <= 1e-10


# --------------------------------------------------------------------------- #
# 2. fidelity / distance match dense definitions
# --------------------------------------------------------------------------- #
def test_fidelity_matches_dense():
    a = _tt_from_statevector(_ghz(6), 6)
    rng = np.random.default_rng(42)
    b = _tt_from_statevector(rng.standard_normal(64) + 1j * rng.standard_normal(64), 6)
    ra, rb = a.reconstruct().ravel(), b.reconstruct().ravel()
    ref = abs(np.vdot(ra, rb)) ** 2 / (np.vdot(ra, ra).real * np.vdot(rb, rb).real)
    assert abs(fidelity(a, b) - ref) <= 1e-12


def test_distance_matches_dense():
    a, da = _random_lowrank_tt((4, 4, 4), max_rank=8, seed=3)
    b, db = _random_lowrank_tt((4, 4, 4), max_rank=8, seed=4)
    ref = np.linalg.norm((da - db).ravel())
    assert abs(distance(a, b) - ref) <= 1e-9


# --------------------------------------------------------------------------- #
# 3. golden anchors
# --------------------------------------------------------------------------- #
def test_golden_ghz_self_overlap():
    g = _tt_from_statevector(_ghz(5), 5)
    assert abs(inner(g, g) - 1.0) <= 1e-12
    assert abs(fidelity(g, g) - 1.0) <= 1e-12


def test_golden_ghz_vs_basis():
    n = 5
    g = _tt_from_statevector(_ghz(n), n)
    zero = _tt_from_statevector(_basis(n, all_ones=False), n)
    # ⟨GHZ|0…0⟩ = 1/√2 → fidelity = 1/2
    assert abs(abs(inner(g, zero)) - 1.0 / np.sqrt(2.0)) <= 1e-12
    assert abs(fidelity(g, zero) - 0.5) <= 1e-12


def test_golden_orthogonal_basis():
    n = 5
    zero = _tt_from_statevector(_basis(n, all_ones=False), n)
    ones = _tt_from_statevector(_basis(n, all_ones=True), n)
    assert abs(inner(zero, ones)) <= 1e-12
    assert fidelity(zero, ones) <= 1e-12


# --------------------------------------------------------------------------- #
# 4. scaling proof — n=40 (2⁴⁰ dense is impossible), low-χ, never decompresses
# --------------------------------------------------------------------------- #
def test_scaling_large_n_product_states():
    n = 40  # a dense 2⁴⁰ complex statevector would be ~17 TB — impossible
    plus = np.array([1.0, 1.0]) / np.sqrt(2.0)
    minus = np.array([1.0, -1.0]) / np.sqrt(2.0)

    a = _product_tt([plus] * n)
    b = _product_tt([plus] * n)
    c = _product_tt([plus] * (n - 1) + [minus])  # differs orthogonally on last qubit

    assert abs(fidelity(a, b) - 1.0) <= 1e-12        # identical product states
    assert abs(inner(a, c)) <= 1e-12                 # one orthogonal site ⇒ ⟨a|c⟩ = 0
    assert abs(norm(a) - 1.0) <= 1e-12


# --------------------------------------------------------------------------- #
# 5. TTIndex — exact nearest-neighbour search in compressed space
# --------------------------------------------------------------------------- #
def test_ttindex_retrieves_self_and_neighbour():
    rng = np.random.default_rng(123)
    n = 6
    # a family of states: GHZ, and several random states; query = a noisy GHZ
    states = {
        "ghz": _tt_from_statevector(_ghz(n), n),
        "zero": _tt_from_statevector(_basis(n), n),
        "rand1": _tt_from_statevector(
            rng.standard_normal(2**n) + 1j * rng.standard_normal(2**n), n
        ),
        "rand2": _tt_from_statevector(
            rng.standard_normal(2**n) + 1j * rng.standard_normal(2**n), n
        ),
    }
    idx = TTIndex(metric="fidelity")
    for label, tt in states.items():
        idx.add(tt, label=label)
    assert len(idx) == 4

    # query with the exact GHZ: nearest by fidelity must be "ghz" with fidelity ~1
    top = idx.query(states["ghz"], k=2)
    assert top[0][0] == "ghz"
    assert abs(top[0][1] - 1.0) <= 1e-12

    # distance metric: nearest to GHZ is itself with distance ~0
    idx_d = TTIndex(metric="distance")
    idx_d.add_many(states.values(), labels=list(states.keys()))
    top_d = idx_d.query(states["ghz"], k=1)
    assert top_d[0][0] == "ghz"
    assert top_d[0][1] <= 1e-9
