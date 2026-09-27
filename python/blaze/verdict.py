"""The Blaze answer: compressed, declined, or undecided.

``compress`` still returns a TT. This is the door that can refuse.
The certificate lives on the return value. BLZ1 is unchanged.
BLZ2 (``blaze.blz2``) stores this certificate and rejects a bad hash.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from .tt import TT, compress, truncation_rank, unfolding_budget


class Kind(str, Enum):
    COMPRESSED = "compressed"
    DECLINED = "declined"
    UNDECIDED = "undecided"


@dataclass(frozen=True)
class Cut:
    index: int
    tol_rank: int
    kept: int
    spectrum_len: int
    delta: float
    cap_bound: bool


@dataclass(frozen=True)
class Verdict:
    kind: Kind
    rel_tol: float
    max_rank: Optional[int]
    shape: tuple[int, ...]
    nparams: int
    dense_nbytes: int
    tt_nbytes: int
    ratio: float
    absolute_bound: float
    relative_bound: float
    measured_rel_error: Optional[float]
    cap_cut: Optional[int]
    cap_tol_rank: Optional[int]
    cap_rank: Optional[int]
    tail_at_cap: Optional[float]
    spectrum: tuple[Cut, ...]
    tt: Optional[TT]


def _tail(singular_values: np.ndarray, kept: int) -> float:
    tail = np.asarray(singular_values)[kept:]
    if tail.size == 0:
        return 0.0
    return float(np.linalg.norm(tail.astype(np.float64)))


def verdict(
    tensor: np.ndarray,
    *,
    rel_tol: float,
    max_rank: Optional[int] = None,
) -> Verdict:
    """INGEST → DECOMPOSE → one of three answers.

    ``rel_tol`` is the declared budget. There is no default.
    Cores are attached only when the kind is ``compressed``.
    """
    if not np.isfinite(rel_tol) or rel_tol < 0.0:
        raise ValueError("rel_tol must be finite and >= 0")
    if max_rank is not None and int(max_rank) < 1:
        raise ValueError("max_rank must be >= 1")
    array = np.asarray(tensor)
    if array.ndim < 2:
        raise ValueError("verdict requires ndim >= 2")

    built = compress(array, max_rank=max_rank, rel_tol=rel_tol, verbose=False)
    eps = unfolding_budget(rel_tol, array.ndim)
    cuts: list[Cut] = []
    for index, singular_values in enumerate(built.singular_values):
        kept, tol_rank, cap_bound = truncation_rank(singular_values, eps, max_rank)
        core_rank = int(built.cores[index].shape[2])
        if kept != core_rank:
            raise RuntimeError(
                f"cut {index}: truncation_rank kept {kept}, core kept {core_rank}"
            )
        cuts.append(
            Cut(
                index=index,
                tol_rank=tol_rank,
                kept=kept,
                spectrum_len=int(np.asarray(singular_values).shape[0]),
                delta=_tail(singular_values, kept),
                cap_bound=cap_bound,
            )
        )

    dense_nbytes = int(array.nbytes)
    tt_nbytes = sum(int(core.nbytes) for core in built.cores)
    ratio = (dense_nbytes / tt_nbytes) if tt_nbytes > 0 else float("inf")
    absolute_bound = float(np.linalg.norm([cut.delta for cut in cuts])) if cuts else 0.0
    frobenius = float(np.linalg.norm(array.ravel()))
    relative_bound = (absolute_bound / frobenius) if frobenius > 0.0 else 0.0

    measured: Optional[float]
    try:
        measured = built.rel_error(array)
    except MemoryError:
        measured = None
    if measured is not None and not np.isfinite(measured):
        measured = None

    binding = [cut for cut in cuts if cut.cap_bound]
    cap_cut = cap_tol_rank = cap_rank = None
    tail_at_cap: Optional[float] = None
    if binding:
        chosen = max(binding, key=lambda cut: (cut.delta, -cut.index))
        cap_cut = chosen.index
        cap_tol_rank = chosen.tol_rank
        cap_rank = int(max_rank) if max_rank is not None else chosen.kept
        tail_at_cap = chosen.delta

    if binding:
        kind = Kind.UNDECIDED
        delivered: Optional[TT] = None
    elif tt_nbytes >= dense_nbytes:
        kind = Kind.DECLINED
        delivered = None
    elif measured is not None and measured <= rel_tol:
        kind = Kind.COMPRESSED
        delivered = built
    else:
        kind = Kind.UNDECIDED
        delivered = None

    return Verdict(
        kind=kind,
        rel_tol=float(rel_tol),
        max_rank=None if max_rank is None else int(max_rank),
        shape=tuple(int(dim) for dim in array.shape),
        nparams=built.nparams(),
        dense_nbytes=dense_nbytes,
        tt_nbytes=tt_nbytes,
        ratio=float(ratio),
        absolute_bound=absolute_bound,
        relative_bound=relative_bound,
        measured_rel_error=measured,
        cap_cut=cap_cut,
        cap_tol_rank=cap_tol_rank,
        cap_rank=cap_rank,
        tail_at_cap=tail_at_cap,
        spectrum=tuple(cuts),
        tt=delivered,
    )
