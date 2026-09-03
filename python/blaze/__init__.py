"""
blaze
High-dimensional data compression via Tensor Train (TT/MPS).

Phase 1 (Python prototype): executable specification of the final API.
See docs/architecture.md (to be updated) and examples/.

Public surface is intentionally small and stable for the Rust port.
"""

from .tt import TT, compress, tt_svd
from .diagnostics import analyze_compressibility, print_compressibility_report
from .overlap import inner, norm, fidelity, distance, TTIndex
from .quantize import QuantizedTT, quantize_tt
from .quantized_search import (
    inner_quantized,
    fidelity_quantized,
    distance_quantized,
    QuantizedTTIndex,
)

__all__ = [
    "TT",
    "compress",
    "tt_svd",
    "analyze_compressibility",
    "print_compressibility_report",
    # Phase 7 — operations in compressed space (no decompression)
    "inner",
    "norm",
    "fidelity",
    "distance",
    "TTIndex",
    # Phase 8 — second-stage core quantization
    "QuantizedTT",
    "quantize_tt",
    # Phase 9 — overlap & search over quantized cores (compressed + quantized)
    "inner_quantized",
    "fidelity_quantized",
    "distance_quantized",
    "QuantizedTTIndex",
]

# cirq integration is optional (import blaze.cirq to trigger the check)
try:
    from . import cirq as cirq  # noqa: F401
except Exception:  # pragma: no cover
    pass

# Version for Phase 1
__version__ = "0.1.0.dev"
