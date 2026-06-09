"""
blaze.quantized_search
Phase 9 — overlap and similarity search over the SECOND-STAGE QUANTIZED cores.

Composes Phase 7 (the MPS zipper, ``blaze.overlap``) with Phase 8 (int8 / low-bit
core codes, ``blaze.quantize``): the indexed database is stored as ``QuantizedTT`` —
typically 4-8x smaller than the float TT — and a query dequantizes each candidate
per-core on the fly, then runs the exact zipper. Retrieval is therefore APPROXIMATE:
the quantization error enters the overlap. This is the calibrated-quantization /
TurboVec analogue — cheap storage, near-exact retrieval *if the recall holds*, which
this phase exists to measure.

Honesty (ADR-0002):
- In NumPy this path DEQUANTIZES per core (there is no int8 ``tensordot``), so the
  win realised *here* is MEMORY (the stored index is int8), NOT FLOPs. The compute
  win needs a genuine int8 kernel — Rust SIMD (AVX-VNNI) or CUDA tensor cores — which
  is Phase 9-Rust/CUDA, deliberately out of scope for this pure-Python reference.
- The error COMPOSES: (TT truncation) + (core quantization), always reported against
  the exact zipper, never hidden. The headline metric is recall@k vs the exact index.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from .tt import TT
from .overlap import inner
from .quantize import QuantizedTT, quantize_tt

__all__ = [
    "inner_quantized",
    "fidelity_quantized",
    "distance_quantized",
    "QuantizedTTIndex",
]


def _as_tt(x) -> TT:
    """Accept either a TT (used as-is) or a QuantizedTT (expanded per-core)."""
    return x.dequantize() if isinstance(x, QuantizedTT) else x


def inner_quantized(a, b) -> complex:
    """⟨a|b⟩ where a and/or b may be a ``QuantizedTT`` (expanded per-core, then zipper).

    Equals ``inner(a.dequantize(), b.dequantize())`` — named so a future int8 kernel
    can replace the body without touching call sites. APPROXIMATE: carries the
    quantization error of whichever argument is quantized.
    """
    return inner(_as_tt(a), _as_tt(b))


def fidelity_quantized(a, b) -> float:
    """State fidelity |⟨a|b⟩|²/(⟨a|a⟩⟨b|b⟩) with quantized operands. Approximate."""
    ta, tb = _as_tt(a), _as_tt(b)
    iab = inner(ta, tb)
    denom = inner(ta, ta).real * inner(tb, tb).real
    return float((abs(iab) ** 2) / denom) if denom > 0.0 else 0.0


def distance_quantized(a, b) -> float:
    """Frobenius/L2 distance ‖a − b‖ with quantized operands. Approximate."""
    ta, tb = _as_tt(a), _as_tt(b)
    na2 = inner(ta, ta).real
    nb2 = inner(tb, tb).real
    re_ab = inner(ta, tb).real
    return float(np.sqrt(max(0.0, na2 + nb2 - 2.0 * re_ab)))


class QuantizedTTIndex:
    """A searchable database whose indexed states are stored QUANTIZED (Phase 8).

    Same query semantics as ``blaze.TTIndex``, but each indexed state is kept as a
    ``QuantizedTT`` (int8 / low-bit codes), so the database footprint is ``nbytes()`` —
    typically 4-8x smaller than the float TT index. At query time each candidate is
    dequantized per-core and scored with the exact zipper; results are approximate and
    Phase 9 measures the recall against the exact ``TTIndex``.

    metric:
      - "fidelity": |⟨q|x⟩|²/(⟨q|q⟩⟨x|x⟩); higher = more similar (default).
      - "distance": ‖q − x‖; lower = more similar.

    [KNOWN_LIMIT] Per query this dequantizes every candidate (O(N·n·d·χ³) float work,
    same as TTIndex) — the only saving in this reference is the stored size. The FLOP
    saving belongs to the int8 kernel (Phase 9-Rust/CUDA). Cheap only for low χ.
    """

    def __init__(self, metric: str = "fidelity") -> None:
        if metric not in ("fidelity", "distance"):
            raise ValueError("metric must be 'fidelity' or 'distance'")
        self.metric = metric
        self._items: list[QuantizedTT] = []
        self._labels: list = []
        self._self2: list[float] = []  # ⟨x|x⟩ on the dequantized state, cached at add

    def __len__(self) -> int:
        return len(self._items)

    def add(self, qtt: QuantizedTT, label=None) -> int:
        """Index one already-quantized state. Returns its integer id."""
        idx = len(self._items)
        x = qtt.dequantize()
        self._items.append(qtt)
        self._labels.append(idx if label is None else label)
        self._self2.append(inner(x, x).real)
        return idx

    def add_tt(self, tt: TT, bits: int = 8, granularity: str = "per_bond", label=None) -> int:
        """Quantize a float TT to ``bits`` (Phase 8) and index it."""
        return self.add(quantize_tt(tt, bits=bits, granularity=granularity), label)

    def add_many(self, qtts: Iterable[QuantizedTT], labels: Sequence | None = None) -> list[int]:
        qtts = list(qtts)
        if labels is None:
            labels = [None] * len(qtts)
        return [self.add(q, lab) for q, lab in zip(qtts, labels)]

    def nbytes(self) -> int:
        """Total stored size of the index — real bit-packed (Phase 8 accounting)."""
        return sum(q.nbytes() for q in self._items)

    def query(self, q, k: int = 5) -> list[tuple]:
        """Return the top-k indexed states most similar to ``q`` (a TT or QuantizedTT).

        Each result is (label, score, index). Candidates are dequantized per-core and
        scored with the exact zipper. Approximate vs the float ``TTIndex``.
        """
        qt = _as_tt(q)
        q2 = inner(qt, qt).real
        scored: list[tuple] = []
        for i, qtt in enumerate(self._items):
            x = qtt.dequantize()
            iqx = inner(qt, x)
            if self.metric == "fidelity":
                denom = q2 * self._self2[i]
                score = float((abs(iqx) ** 2) / denom) if denom > 0.0 else 0.0
            else:  # distance
                d2 = q2 + self._self2[i] - 2.0 * iqx.real
                score = float(np.sqrt(max(0.0, d2)))
            scored.append((self._labels[i], score, i))
        scored.sort(key=lambda row: row[1], reverse=(self.metric == "fidelity"))
        return scored[:k]
