"""
blaze
High-dimensional data compression via Tensor Train (TT/MPS).

Fase 1 (Python prototype): executable specification of the final API.
See docs/architecture.md (to be updated) and examples/.

Public surface is intentionally small and stable for the Rust port.
"""

from .tt import TT, compress, tt_svd
from .diagnostics import analyze_compressibility, print_compressibility_report

__all__ = [
    "TT",
    "compress",
    "tt_svd",
    "analyze_compressibility",
    "print_compressibility_report",
]

# cirq integration is optional (import blaze.cirq to trigger the check)
try:
    from . import cirq as cirq  # noqa: F401
except Exception:  # pragma: no cover
    pass

# Version for Fase 1
__version__ = "0.1.0.dev"
