"""
blaze.tt
Tensor Train (TT / MPS) compression via TT-SVD.

Phase 1 pure-Python prototype. Mirrors the planned Rust/CUDA API surface.

Dtypes: float32, float64, complex128 (c64 deferred per design).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.linalg import svd


def _as_float_or_complex(arr: np.ndarray) -> np.ndarray:
    """Ensure a supported dtype. We keep user's precision when possible."""
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        return arr
    if arr.dtype == np.complex64:
        # Defer c64: upcast for stability in Phase 1
        return arr.astype(np.complex128)
    if arr.dtype == np.complex128:
        return arr
    # fallback: promote real to f64, complex to c128
    if np.iscomplexobj(arr):
        return arr.astype(np.complex128)
    return arr.astype(np.float64)


@dataclass
class TT:
    """Tensor Train representation.

    cores: list of ndarrays, each of shape (r_left, d_phys, r_right).
           r_left[0] == 1, r_right[-1] == 1.
    shape: original tensor shape (d0, d1, ..., d_{n-1}).
    singular_values: list of 1D arrays (length ndim-1). The *full* singular values
                     from each successive matricization (before truncation).
                     Useful for decay diagnostics and effective rank.
    """
    cores: list[np.ndarray]
    shape: tuple[int, ...]
    singular_values: list[np.ndarray]

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def ranks(self) -> list[int]:
        """Bond dimensions: [1, r0, r1, ..., rn-2, 1]"""
        rs = [1]
        for c in self.cores:
            rs.append(int(c.shape[2]))
        return rs

    def nparams(self) -> int:
        """Total number of stored parameters (real or complex scalars)."""
        return sum(int(c.size) for c in self.cores)

    def dtype(self) -> np.dtype:
        return self.cores[0].dtype if self.cores else np.dtype("float64")

    def reconstruct(self) -> np.ndarray:
        """Contract the TT back to a dense tensor of self.shape.

        For Phase 1 (small tensors) this materializes the full array.
        """
        if not self.cores:
            return np.zeros(self.shape, dtype=self.dtype())

        ndim = self.ndim
        if ndim == 1:
            # edge case: single core (1, d0, 1) -> (d0,)
            return self.cores[0].squeeze().reshape(self.shape)

        # Successive contraction.
        # After loop: res has shape (1, d0, d1, ..., d_{n-1}, 1)
        res = self.cores[0]
        for core in self.cores[1:]:
            res = np.tensordot(res, core, axes=([res.ndim - 1], [0]))
        # Drop the two singleton bond dimensions
        res = np.squeeze(res, axis=(0, -1))
        return res.reshape(self.shape)

    def rel_error(self, original: np.ndarray) -> float:
        """Relative Frobenius error: ||T - approx||_F / ||T||_F ."""
        approx = self.reconstruct()
        orig = original.astype(approx.dtype, copy=False)
        num = np.linalg.norm((orig - approx).ravel())
        den = np.linalg.norm(orig.ravel())
        return float(num / den) if den > 0 else 0.0

    def compression_ratio(self, original: np.ndarray) -> float:
        """Raw byte ratio (original.nbytes / TT.nbytes).

        This is the storage compression (ignores headers, alignment, etc.).
        Honest for comparing against baselines.
        """
        orig_bytes = int(original.nbytes)
        tt_bytes = sum(int(c.nbytes) for c in self.cores)
        return (orig_bytes / tt_bytes) if tt_bytes > 0 else float("inf")

    def effective_ranks(self, tol: float = 1e-4) -> list[int]:
        """Minimal bond dimension per unfolding to capture (1 - tol^2) energy.

        This is the direct classical analogue of "entanglement rank" / Schmidt rank
        for a given precision. If these numbers stay small across bonds, the
        tensor is TT-compressible. If they stay close to min(dim_left, dim_right),
        the data has no low-rank TT structure along this mode ordering.
        """
        eff: list[int] = []
        for s in self.singular_values:
            if s.size == 0:
                eff.append(0)
                continue
            s2 = s.astype(np.float64) ** 2
            total = s2.sum()
            if total <= 0:
                eff.append(1)
                continue
            cum = np.cumsum(s2) / total
            # smallest r such that cum[r-1] >= 1 - tol**2
            idx = int(np.searchsorted(cum, 1.0 - tol * tol))
            r = min(idx + 1, len(s))
            eff.append(max(r, 1))
        return eff

    def singular_decay_summary(self, tols: Sequence[float] = (1e-2, 1e-4, 1e-6, 1e-8)) -> dict:
        """For each bond, report effective rank at several tolerances.

        Returns a dict suitable for printing / logging. Makes the
        [KNOWN_LIMIT] "is this data TT-compressible at all?" measurable.
        """
        summary = {}
        for b, s in enumerate(self.singular_values):
            if s.size == 0:
                continue
            s2 = s.astype(np.float64) ** 2
            total = s2.sum()
            if total <= 0:
                summary[b] = {t: 1 for t in tols}
                continue
            cum = np.cumsum(s2) / total
            per_tol = {}
            for t in tols:
                idx = int(np.searchsorted(cum, 1.0 - t * t))
                per_tol[t] = min(idx + 1, len(s))
            summary[b] = per_tol
        return summary


def tt_svd(
    tensor: np.ndarray,
    max_rank: Optional[int] = None,
    rel_tol: float = 1e-4,
    verbose: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Compute a TT decomposition via successive SVD + truncation (TT-SVD).

    This is the workhorse for Phase 1. The algorithm is deterministic and the
    truncation error is controlled per unfolding (Frobenius energy on the matricization).

    Returns:
        cores, singular_values (full S arrays from each of the ndim-1 unfoldings)
    """
    tensor = _as_float_or_complex(tensor)
    shape = tuple(int(x) for x in tensor.shape)
    ndim = len(shape)
    if ndim == 0:
        raise ValueError("scalar tensor not supported")
    if ndim == 1:
        # Represent as (1, d0, 1)
        core = tensor.reshape(1, shape[0], 1)
        return [core], []

    cores: list[np.ndarray] = []
    singular_values: list[np.ndarray] = []

    # remaining starts as the full tensor; we progressively factor left cores
    # and keep the "right factor" as a tensor of shape (r, d_{k+1}, ...)
    remaining = tensor
    r_prev = 1

    # Global energy for optional global tol interpretation (we still do per-step)
    # For TT-SVD the per-matricization relative tail is the practical control.
    eps = rel_tol / max(1.0, np.sqrt(ndim - 1))  # distribute budget roughly

    for i in range(ndim - 1):
        d_i = shape[i]
        if i == 0:
            # (d0, prod_rest)
            mat = remaining.reshape(d_i, -1)
        else:
            # remaining.shape == (r_prev, d_i, d_{i+1}, ...)
            mat = remaining.reshape(r_prev * d_i, -1)

        # Full SVD (U: left, S real, Vh)
        U, S, Vh = svd(mat, full_matrices=False)

        # Record full spectrum for diagnostics (the key for claim (b) honesty)
        singular_values.append(S.copy())

        # Truncation: energy on this unfolding + max_rank cap
        s2 = S.astype(np.float64) ** 2
        total_s2 = s2.sum()
        if total_s2 > 0 and eps > 0:
            cum = np.cumsum(s2) / total_s2
            # first index where we have captured enough
            idx = int(np.searchsorted(cum, 1.0 - eps * eps))
            r_tol = min(idx + 1, len(S))
        else:
            r_tol = len(S)

        r = r_tol
        if max_rank is not None:
            r = min(r, int(max_rank))
        r = max(r, 1)
        r = min(r, len(S))

        if verbose:
            kept_energy = (s2[:r].sum() / total_s2) if total_s2 > 0 else 1.0
            print(f"[tt_svd] mode {i}: rank {r}/{len(S)}  "
                  f"kept_energy={kept_energy:.6e}  max_sigma={S[0]:.3e}")

        U = U[:, :r]
        S = S[:r]
        Vh = Vh[:r, :]

        # Build core
        if i == 0:
            core = U.reshape(1, d_i, r)
        else:
            core = U.reshape(r_prev, d_i, r)
        cores.append(core)

        # Next remaining = diag(S) @ Vh , viewed as (r, d_{i+1}, ...)
        remaining = (S[:, None] * Vh).reshape((r,) + shape[i + 1 :])
        r_prev = r

    # Final core (r_prev, d_last, 1)
    last = remaining.reshape(r_prev, shape[-1], 1)
    cores.append(last)

    return cores, singular_values


def compress(
    tensor: np.ndarray,
    max_rank: Optional[int] = None,
    rel_tol: float = 1e-4,
    verbose: bool = False,
) -> TT:
    """High-level entry point. Mirrors the future Rust API.

    tensor: np.ndarray (any ndim >= 2, real or complex128/64, f32/f64)
    max_rank: hard cap on every bond dimension
    rel_tol: target relative Frobenius error (per-unfolding energy budget distributed)

    The returned TT can be reconstructed, measured for error, etc.
    """
    if tensor.ndim < 2:
        raise ValueError("compress requires ndim >= 2 (for ndim=1 use trivial TT)")

    cores, sigmas = tt_svd(
        tensor,
        max_rank=max_rank,
        rel_tol=rel_tol,
        verbose=verbose,
    )
    return TT(cores=cores, shape=tensor.shape, singular_values=sigmas)
