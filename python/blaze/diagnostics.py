"""
blaze.diagnostics

Tools to diagnose whether a tensor is meaningfully TT-compressible
*before* or *after* compression. This turns the architectural [KNOWN_LIMIT]
("quality depends on the structure of the data") into a measurable quantity.

Reuses the classical analogue of entanglement spectrum / singular value decay
from quantum MPS analysis (SUBSTRATE muscle).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .tt import TT, compress, tt_svd


def analyze_compressibility(
    tensor: np.ndarray,
    max_rank: int | None = 32,
    rel_tol: float = 1e-4,
    tols: Sequence[float] = (1e-2, 1e-4, 1e-6, 1e-8),
) -> dict:
    """Run a cheap diagnostic compression pass and report decay behavior.

    Returns a dict with:
      - 'effective_ranks': list per bond for the given rel_tol
      - 'decay_summary': effective rank needed at multiple tols
      - 'max_possible_ranks': min(left_dim, right_dim) for each unfolding
      - 'recommendation': crude text signal ("highly compressible", "marginal", "not TT-friendly")
      - 'achieved_rel_error': if we did a full compress+reconstruct

    This is the tool that tells you *before heavy investment* whether claim (b)
    (classical high-order data) has a chance.
    """
    tensor = np.asarray(tensor)
    ndim = tensor.ndim
    if ndim < 2:
        return {"error": "ndim < 2"}

    # Do the decomposition (we only need the singular_values; we can avoid full TT if wanted)
    _, sigmas = tt_svd(tensor, max_rank=max_rank, rel_tol=rel_tol, verbose=False)

    # Build a temporary TT just for the helpers
    # (we re-compress with same params for error + ranks)
    tt = compress(tensor, max_rank=max_rank, rel_tol=rel_tol, verbose=False)

    eff = tt.effective_ranks(rel_tol)
    decay = tt.singular_decay_summary(tols)

    # Theoretical max rank per bond (for the chosen mode ordering)
    max_ranks = []
    prod_left = 1
    for i, d in enumerate(tensor.shape[:-1]):
        prod_left *= d
        prod_right = int(np.prod(tensor.shape[i + 1 :]))
        max_ranks.append(min(prod_left, prod_right))

    # Crude but honest recommendation (relaxed for small physical dims common in practice)
    median_eff = float(np.median(eff)) if eff else 0.0
    worst_eff = max(eff) if eff else 0
    ratio_to_max = [e / max(1, m) for e, m in zip(eff, max_ranks)]
    max_ratio = max(ratio_to_max) if ratio_to_max else 0.0

    if worst_eff <= 6 or (median_eff <= 6 and max_ratio < 0.4):
        rec = "highly TT-compressible (strong low-rank structure along chain)"
    elif worst_eff <= 16 or median_eff <= 12:
        rec = "moderately compressible; good candidate for TT (inspect decay + compare to SVD baseline)"
    else:
        rec = "high effective ranks (data may not be TT-friendly for this mode ordering; try reordering or accept higher ranks / different method)"

    achieved_err = tt.rel_error(tensor)

    return {
        "ndim": ndim,
        "shape": tensor.shape,
        "effective_ranks_at_tol": eff,
        "decay_at_tols": decay,
        "max_ranks_per_bond": max_ranks,
        "median_effective_rank": median_eff,
        "worst_effective_rank": worst_eff,
        "achieved_rel_fro_error": achieved_err,
        "recommendation": rec,
        "note": "Low effective ranks + fast singular decay => classical data (b) is TT-friendly. Flat spectrum => use matrix methods or accept high rank.",
    }


def print_compressibility_report(tensor: np.ndarray, **kwargs) -> None:
    """Pretty-print the analysis for notebooks / CLI."""
    info = analyze_compressibility(tensor, **kwargs)
    print("=== Blaze TT Compressibility Diagnostic ===")
    print(f"shape: {info.get('shape')}  ndim={info.get('ndim')}")
    print(f"effective ranks (tol={kwargs.get('rel_tol', 1e-4)}): {info.get('effective_ranks_at_tol')}")
    print(f"max ranks (by unfolding): {info.get('max_ranks_per_bond')}")
    print(f"achieved rel. Fro error: {info.get('achieved_rel_fro_error'):.3e}")
    print(f"recommendation: {info.get('recommendation')}")
    print("decay summary (effective rank needed for each tol):")
    for bond, per_tol in info.get("decay_at_tols", {}).items():
        print(f"  bond {bond}: {per_tol}")
    print("=============================================")
