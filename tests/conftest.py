"""Pytest bootstrap.

Ensure tests import the in-tree ``blaze`` package (with examples/ and future
adapters/) rather than a possibly-stale installed copy. Mirrors the
``sys.path`` setup the runnable examples do for themselves.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "python"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
