"""BLZ2 roundtrip. A flipped byte is rejected. A different tensor is rejected."""

from pathlib import Path

import numpy as np
import pytest

from blaze import Kind, verdict
from blaze.blz2 import BlzError, read_verdict, verify_verdict, write_verdict


def _separable() -> np.ndarray:
    rng = np.random.default_rng(1)
    factors = [rng.standard_normal(4) for _ in range(3)]
    tensor = factors[0][:, None, None] * factors[1][None, :, None] * factors[2][None, None, :]
    return tensor.astype(np.float64)


def _haar(n: int = 6) -> np.ndarray:
    rng = np.random.default_rng(0)
    values = rng.standard_normal(2**n) + 1j * rng.standard_normal(2**n)
    values /= np.linalg.norm(values)
    return values.reshape((2,) * n).astype(np.complex128)


def test_compressed_roundtrip_and_recompute(tmp_path: Path):
    tensor = _separable()
    answer = verdict(tensor, rel_tol=1e-8)
    assert answer.kind is Kind.COMPRESSED
    path = tmp_path / "sep.blz2"
    write_verdict(path, answer)
    stored = read_verdict(path)
    assert stored.kind is Kind.COMPRESSED
    assert stored.tt is not None
    assert stored.shape == tensor.shape
    assert stored.nparams == answer.nparams
    checked = verify_verdict(path, tensor)
    assert checked.kind is Kind.COMPRESSED
    other = np.random.default_rng(2).standard_normal(tensor.shape)
    with pytest.raises(BlzError):
        verify_verdict(path, other)


def test_flipped_byte_is_rejected(tmp_path: Path):
    tensor = _separable()
    path = tmp_path / "sep.blz2"
    write_verdict(path, verdict(tensor, rel_tol=1e-8))
    raw = bytearray(path.read_bytes())
    raw[20] ^= 0xFF
    path.write_bytes(raw)
    with pytest.raises(BlzError, match="hash"):
        read_verdict(path)


def test_declined_has_no_cores(tmp_path: Path):
    tensor = _haar(6)
    answer = verdict(tensor, rel_tol=1e-6)
    assert answer.kind is Kind.DECLINED
    assert answer.tt is None
    path = tmp_path / "haar.blz2"
    write_verdict(path, answer)
    stored = read_verdict(path)
    assert stored.kind is Kind.DECLINED
    assert stored.tt is None
    assert stored.ratio < 1.0
    verify_verdict(path, tensor)
