"""
blaze.overlap
Phase 7 — inner products, norms, fidelity, distance and similarity search computed
DIRECTLY on the compressed TT/MPS representation, **never decompressing**.

The primitive is the MPS "zipper" contraction. The overlap ⟨A|B⟩ contracts the two
trains site by site through a transfer matrix E of shape (r_A, r_B):

    E ← [[1]]                                      # (1,1)
    for A_k, B_k in zip(a.cores, b.cores):         # A is the bra (conjugated)
        T ← tensordot(E,        B_k, [[1],[0]])    # (aL, d, bR)
        E ← tensordot(conj(A_k), T,  [[0,1],[0,1]])# (aR, bR)
    ⟨A|B⟩ ← E[0,0]

Cost is O(n · d · χ³) instead of O(dⁿ) for the dense inner product. For low-bond
(compressible) states this is the difference between instant and impossible.

This is the differential primitive (ADR-0002): exact similarity search over
compressed states WITHOUT decompression — stronger than approximate (PQ-style)
vector quantization, *when the data is genuinely TT-structured*.

Honesty (ADR-0001 / ADR-0002): the cost is O(nχ³); it is cheap ONLY for low χ.
Volume-law states (χ = d^{n/2}) cost as much as the dense product — documented, not
hidden. There is no advantage claim beyond the measured exactness + scaling.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from .tt import TT

__all__ = ["inner", "norm", "fidelity", "distance", "TTIndex"]


def _check_compatible(a: TT, b: TT) -> None:
    if tuple(a.shape) != tuple(b.shape):
        raise ValueError(
            f"overlap requires identical physical shape, got {a.shape} vs {b.shape}"
        )
    if len(a.cores) != len(b.cores):
        raise ValueError(
            f"overlap requires the same number of cores, got {len(a.cores)} vs {len(b.cores)}"
        )


def inner(a: TT, b: TT) -> complex:
    """⟨a|b⟩ via the MPS zipper contraction. The bra `a` is conjugated.

    Equals ``np.vdot(a.reconstruct().ravel(), b.reconstruct().ravel())`` exactly
    (to floating-point), but never materializes the dense tensor. O(n·d·χ³).
    """
    _check_compatible(a, b)
    # Transfer matrix E: (r_a_left, r_b_left). Starts as the 1×1 boundary.
    e = np.ones((1, 1), dtype=np.complex128)
    for core_a, core_b in zip(a.cores, b.cores):
        ca = core_a.astype(np.complex128, copy=False)
        cb = core_b.astype(np.complex128, copy=False)
        # T[aL, d, bR] = Σ_bL E[aL, bL] · B[bL, d, bR]
        t = np.tensordot(e, cb, axes=([1], [0]))          # (aL, d, bR)
        # E[aR, bR] = Σ_{aL, d} conj(A[aL, d, aR]) · T[aL, d, bR]
        e = np.tensordot(ca.conj(), t, axes=([0, 1], [0, 1]))  # (aR, bR)
    return complex(e.reshape(()))


def norm(a: TT) -> float:
    """‖a‖ = √(Re⟨a|a⟩), computed on the TT directly."""
    return float(np.sqrt(max(0.0, inner(a, a).real)))


def fidelity(a: TT, b: TT) -> float:
    """State fidelity |⟨a|b⟩|² / (⟨a|a⟩·⟨b|b⟩) ∈ [0, 1], normalization-free.

    The no-decompression replacement for the dense ``blaze.cirq.fidelity``.
    """
    iab = inner(a, b)
    denom = inner(a, a).real * inner(b, b).real
    if denom <= 0.0:
        return 0.0
    return float((abs(iab) ** 2) / denom)


def distance(a: TT, b: TT) -> float:
    """Frobenius/L2 distance ‖a − b‖ between the represented tensors, via overlaps.

    Uses ‖a−b‖² = ⟨a|a⟩ + ⟨b|b⟩ − 2 Re⟨a|b⟩. Never forms (a − b) densely.
    """
    na2 = inner(a, a).real
    nb2 = inner(b, b).real
    re_ab = inner(a, b).real
    return float(np.sqrt(max(0.0, na2 + nb2 - 2.0 * re_ab)))


class TTIndex:
    """A searchable database of compressed TT/MPS states.

    Similarity search runs ENTIRELY in compressed space via the overlap zipper —
    indexed states are never decompressed. The search is **exact** (not approximate,
    unlike PQ-style vector quantization) for TT-structured data.

    metric:
      - "fidelity": |⟨q|x⟩|²/(⟨q|q⟩⟨x|x⟩); higher = more similar (default).
      - "distance": ‖q − x‖; lower = more similar.

    [KNOWN_LIMIT] Each query is O(N · n · d · χ³) for N indexed states. This is cheap
    only while χ stays small (compressible / low-entanglement data). It is NOT a
    general-purpose vector database: generic high-entropy embeddings are not
    TT-structured and will not be cheap here (ADR-0001/0002).
    """

    def __init__(self, metric: str = "fidelity") -> None:
        if metric not in ("fidelity", "distance"):
            raise ValueError("metric must be 'fidelity' or 'distance'")
        self.metric = metric
        self._items: list[TT] = []
        self._labels: list = []
        self._self2: list[float] = []  # cached ⟨x|x⟩ for each indexed state

    def __len__(self) -> int:
        return len(self._items)

    def add(self, tt: TT, label=None) -> int:
        """Index one compressed state. Returns its integer id."""
        idx = len(self._items)
        self._items.append(tt)
        self._labels.append(idx if label is None else label)
        self._self2.append(inner(tt, tt).real)
        return idx

    def add_many(self, tts: Iterable[TT], labels: Sequence | None = None) -> list[int]:
        tts = list(tts)
        if labels is None:
            labels = [None] * len(tts)
        return [self.add(tt, lab) for tt, lab in zip(tts, labels)]

    def query(self, q: TT, k: int = 5) -> list[tuple]:
        """Return the top-k most similar indexed states to `q`.

        Each result is (label, score, index). Scores are fidelity (descending) or
        distance (ascending) per `self.metric`. Computed without decompressing.
        """
        q2 = inner(q, q).real
        scored: list[tuple] = []
        for i, x in enumerate(self._items):
            iqx = inner(q, x)
            if self.metric == "fidelity":
                denom = q2 * self._self2[i]
                score = float((abs(iqx) ** 2) / denom) if denom > 0.0 else 0.0
            else:  # distance
                d2 = q2 + self._self2[i] - 2.0 * iqx.real
                score = float(np.sqrt(max(0.0, d2)))
            scored.append((self._labels[i], score, i))
        scored.sort(key=lambda row: row[1], reverse=(self.metric == "fidelity"))
        return scored[:k]
