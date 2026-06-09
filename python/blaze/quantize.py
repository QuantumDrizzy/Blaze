"""
blaze.quantize
Phase 8 — second-stage compression: quantize the TT core ENTRIES to low-bit codes.

TT-SVD (Phase 2) compresses STRUCTURE via rank truncation. This orthogonal stage
compresses the residual NUMERIC CONTENT of the cores: each core entry → a b-bit
signed integer code + a shared scale (per-core or per-bond). Dequantization is
code·scale. This is the calibrated-scalar-quantization idea (the vector-search
line: store cheap codes, not the raw numbers) applied to TT cores.

The two stages are orthogonal and their errors COMPOSE: the combined relative error
is bounded by (truncation error) + (quantization error) and always reported against
the dense original — never hidden, never compared to a lossless codec.

Honest accounting (ADR-0002):
- storage is the REAL bit-packed size: ceil(n_codes·bits / 8) + scale bytes. The
  reference keeps int8 code arrays for clarity, but `nbytes()` reports the packed
  size a `.blz` writer would actually use; the quantization error is identical
  whether or not the nibbles are physically packed.
- complex128 entries are quantized as separate real/imag codes.

[KNOWN_LIMIT] For very small cores the per-core scale metadata is non-negligible;
for aggressive bit widths (≤4) the state fidelity drops measurably. Both are
measured and reported (see tests + docs/PHASE8-results.md), not hidden.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .tt import TT

__all__ = ["QuantizedTT", "quantize_tt"]


# --------------------------------------------------------------------------- #
# affine symmetric quantization of a real core (r_l, d, r_r)
# --------------------------------------------------------------------------- #
def _code_dtype(bits: int):
    """Smallest signed int that holds ±(2^(bits-1)-1). The reference in-memory dtype;
    `nbytes()` always reports the bit-PACKED size, independent of this."""
    return np.int8 if bits <= 8 else np.int16


def _quantize_real(arr: np.ndarray, bits: int, granularity: str):
    """Return (codes (r_l,d,r_r), scales:float32).

    Symmetric affine: code = round(x / scale) clipped to ±(2^(bits-1)-1). Scales are
    stored float32 — the scale precision is far finer than the 1/qmax code step, so
    this is lossless in practice and halves the metadata vs float64.
    per_core  → one scale (shape (1,));  per_bond → one scale per right-bond column.
    """
    qmax = (1 << (bits - 1)) - 1  # 127 @8b, 31 @6b, 7 @4b, 8191 @14b
    idt = _code_dtype(bits)
    if granularity == "per_core":
        amax = float(np.max(np.abs(arr))) if arr.size else 0.0
        scale = amax / qmax if amax > 0.0 else 1.0
        codes = np.clip(np.round(arr / scale), -qmax, qmax).astype(idt)
        return codes, np.array([scale], dtype=np.float32)

    if granularity == "per_bond":
        r_l, d, r_r = arr.shape
        mat = arr.reshape(r_l * d, r_r)
        amax = np.max(np.abs(mat), axis=0) if mat.size else np.zeros(r_r)  # (r_r,)
        scales = np.where(amax > 0.0, amax / qmax, 1.0).astype(np.float32)
        codes = np.clip(np.round(mat / scales[None, :]), -qmax, qmax).astype(idt)
        return codes.reshape(r_l, d, r_r), scales

    raise ValueError("granularity must be 'per_core' or 'per_bond'")


def _dequantize_real(codes: np.ndarray, scales: np.ndarray, granularity: str) -> np.ndarray:
    s = scales.astype(np.float64)
    if granularity == "per_core":
        return codes.astype(np.float64) * s[0]
    r_l, d, r_r = codes.shape
    mat = codes.reshape(r_l * d, r_r).astype(np.float64) * s[None, :]
    return mat.reshape(r_l, d, r_r)


# --------------------------------------------------------------------------- #
# QuantizedTT
# --------------------------------------------------------------------------- #
@dataclass
class QuantizedTT:
    """A TT whose core entries are stored as b-bit integer codes + scales.

    Reconstruct a usable (complex128/float64) TT with `dequantize()`. Storage is
    reported by `nbytes()` as the real bit-packed size.
    """

    codes_re: list[np.ndarray]
    scales_re: list[np.ndarray]
    codes_im: list[np.ndarray] | None
    scales_im: list[np.ndarray] | None
    shape: tuple[int, ...]
    bits: int
    granularity: str

    @property
    def is_complex(self) -> bool:
        return self.codes_im is not None

    def dequantize(self) -> TT:
        """Rebuild a TT (singular_values dropped — no longer meaningful after quant)."""
        cores: list[np.ndarray] = []
        for i in range(len(self.codes_re)):
            re = _dequantize_real(self.codes_re[i], self.scales_re[i], self.granularity)
            if self.is_complex:
                im = _dequantize_real(self.codes_im[i], self.scales_im[i], self.granularity)
                cores.append((re + 1j * im).astype(np.complex128))
            else:
                cores.append(re.astype(np.float64))
        return TT(cores=cores, shape=tuple(self.shape), singular_values=[])

    def nbytes(self) -> int:
        """Real packed storage: ceil(n_codes·bits/8) payload + float64 scales.

        (A tiny shape/ranks header — a few dozen bytes — is omitted, exactly as
        TT.compression_ratio counts only core bytes. It is negligible and identical
        across schemes.)
        """
        n_codes = sum(c.size for c in self.codes_re)
        n_scales = sum(s.size for s in self.scales_re)
        if self.is_complex:
            n_codes *= 2
            n_scales *= 2
        code_bytes = math.ceil(n_codes * self.bits / 8)
        scale_bytes = n_scales * 4  # float32 scales
        return code_bytes + scale_bytes

    def compression_ratio_over_tt(self, tt: TT) -> float:
        """How much smaller than the (already TT-compressed) source cores."""
        tt_bytes = sum(c.nbytes for c in tt.cores)
        q = self.nbytes()
        return (tt_bytes / q) if q > 0 else float("inf")

    def compression_ratio_over_dense(self, original_nbytes: int) -> float:
        """Total ratio vs the dense original tensor (the headline, end-to-end)."""
        q = self.nbytes()
        return (original_nbytes / q) if q > 0 else float("inf")


def quantize_tt(tt: TT, bits: int = 8, granularity: str = "per_bond") -> QuantizedTT:
    """Quantize every core entry of `tt` to `bits`-bit codes (default per-bond scales).

    bits ∈ {2,4,6,8,...}; granularity ∈ {'per_core','per_bond'}. Complex cores are
    quantized as separate real/imag codes.
    """
    if bits < 2 or bits > 16:
        raise ValueError("bits must be in [2, 16]")
    is_complex = np.iscomplexobj(tt.cores[0]) if tt.cores else False

    codes_re: list[np.ndarray] = []
    scales_re: list[np.ndarray] = []
    codes_im: list[np.ndarray] | None = [] if is_complex else None
    scales_im: list[np.ndarray] | None = [] if is_complex else None

    for core in tt.cores:
        if is_complex:
            cr, sr = _quantize_real(np.ascontiguousarray(core.real), bits, granularity)
            ci, si = _quantize_real(np.ascontiguousarray(core.imag), bits, granularity)
            codes_re.append(cr); scales_re.append(sr)
            codes_im.append(ci); scales_im.append(si)
        else:
            cr, sr = _quantize_real(np.ascontiguousarray(core), bits, granularity)
            codes_re.append(cr); scales_re.append(sr)

    return QuantizedTT(
        codes_re=codes_re, scales_re=scales_re,
        codes_im=codes_im, scales_im=scales_im,
        shape=tuple(tt.shape), bits=bits, granularity=granularity,
    )
