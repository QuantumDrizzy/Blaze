"""Pytest bootstrap.

Ensure tests import the in-tree ``blaze`` package (with examples/ and future
adapters/) rather than a possibly-stale installed copy. Mirrors the
``sys.path`` setup the runnable examples do for themselves.
"""

import os
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "python"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# Ecosystem tests run Blaze against its sibling checkouts: a QuBLAR build (its ROI output), LYTH and
# Labare (the Unibit emulator). Where those are absent -- any machine but the author's, and CI -- they
# skip with the missing pieces named, never silently. Point $BLAZE_ECOSYSTEM / $LABARE_DIR at them.
_ROOT = Path(os.environ.get("BLAZE_ECOSYSTEM", Path(__file__).resolve().parents[4]))
_LABARE = Path(os.environ.get("LABARE_DIR", Path.home() / "Desktop" / "Labare"))
_NEEDS = {
    "QuBLAR build (build/ising_out_roi.txt)": _ROOT / "PR0JECTS" / "RESEARCH" / "QuBLAR" / "build" / "ising_out_roi.txt",
    "LYTH checkout": _ROOT / "LYTH" / "Cargo.toml",
    "Labare checkout": _LABARE / "Cargo.toml",
}


def pytest_configure(config):
    config.addinivalue_line("markers", "ecosystem: needs the sibling QuBLAR, LYTH and Labare checkouts")


def pytest_collection_modifyitems(config, items):
    missing = [name for name, path in _NEEDS.items() if not path.is_file()]
    if not missing:
        return
    skip = pytest.mark.skip(reason="ecosystem test; missing: " + ", ".join(missing))
    for item in items:
        if "ecosystem" in item.keywords:
            item.add_marker(skip)
