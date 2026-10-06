"""QuBLAR ghost bits through Blaze, then one ZIPPER2 source on both machines.

ADR-0002 item 2. A verdict that is not compressed does not produce cores and
does not run. χ = 2 is the word: a compressed train whose bond exceeds 2 does
not run either. The ROI marginals are the published check: int8 cores against
the exact posterior, max |err| at most 2.7e-3 (RESULTS-phase6).
"""

from __future__ import annotations

import importlib.util
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from blaze import Kind, quantize_tt, verdict

# The sibling checkouts (QuBLAR, LYTH): $BLAZE_ECOSYSTEM, else the folder that holds them here.
DESKTOP = Path(os.environ.get("BLAZE_ECOSYSTEM", Path(__file__).resolve().parents[4]))
ROI = DESKTOP / "PR0JECTS" / "RESEARCH" / "QuBLAR" / "build" / "ising_out_roi.txt"
LYTH = DESKTOP / "LYTH"
QGPU = Path(__file__).resolve().parent / "test_qgpu_zipper.py"
BRANCHES = DESKTOP / "PR0JECTS" / "RESEARCH" / "QuBLAR" / "tools" / "branches_blaze.py"

REL_TOL = 1e-6
# Filed in QuBLAR RESULTS-phase6. Not a new bar.
INT8_VS_EXACT = 2.7e-3


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _u64s(lanes: list[int]) -> str:
    return " ".join(f"0x{lane & 0xFFFFFFFFFFFFFFFF:016X}" for lane in lanes)


def _set_u8(lanes: list[int], idx: int, val: int) -> None:
    lane = (idx >> 3) & 3
    byte = idx & 7
    mask = ~(0xFF << (byte * 8)) & 0xFFFFFFFFFFFFFFFF
    lanes[lane] = (lanes[lane] & mask) | ((val & 0xFF) << (byte * 8))


def _set_u32(lanes: list[int], idx: int, bits: int) -> None:
    lane = (idx >> 1) & 3
    half = idx & 1
    mask = ~(0xFFFFFFFF << (half * 32)) & 0xFFFFFFFFFFFFFFFF
    lanes[lane] = (lanes[lane] & mask) | ((bits & 0xFFFFFFFF) << (half * 32))


def _ones_bra() -> list[int]:
    """Sum over the physical index. Code 1 times scale 1.0, not 127/127."""
    lanes = [0, 0, 0, 0]
    one = struct.unpack("<I", struct.pack("<f", np.float32(1.0)))[0]
    _set_u32(lanes, 4, one)
    _set_u32(lanes, 5, one)
    _set_u8(lanes, 0, 1)
    _set_u8(lanes, 4, 1)
    return lanes


def _zero_phys0(lanes: list[int]) -> list[int]:
    out = list(lanes)
    for left in range(2):
        for right in range(2):
            idx = 2 * ((2 * left) * 2 + right)
            _set_u8(out, idx, 0)
            _set_u8(out, idx + 1, 0)
    return out


def _steps(answer):
    """Packed chains, or None when the verdict refuses the machine."""
    if answer.kind is not Kind.COMPRESSED or answer.tt is None:
        return None
    if max(answer.tt.ranks) > 2:
        return None
    qgpu = _load(QGPU, "qgpu_zipper")
    deq = quantize_tt(answer.tt, bits=8, granularity="per_bond").dequantize()
    packed = [qgpu._lanes_from_core(qgpu._pad(np.asarray(core))) for core in deq.cores]
    bra = _ones_bra()
    unpacked_bra = qgpu._unpack(bra)
    assert unpacked_bra[0, 0, 0] == np.float32(1.0)
    assert unpacked_bra[0, 1, 0] == np.float32(1.0)
    chains = []
    for project in [None, *range(len(packed))]:
        steps = []
        for index, lanes in enumerate(packed):
            ket = _zero_phys0(lanes) if project == index else list(lanes)
            steps.append((ket, list(bra)))
        chains.append(steps)
    return deq, packed, bra, chains


def _write_fixture(path: Path, chains) -> None:
    lines = [str(len(chains))]
    for chain in chains:
        lines.append(str(len(chain)))
        for ket, bra in chain:
            lines.append(_u64s(ket))
            lines.append(_u64s(bra))
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _run_machines(chains, tmp_path: Path) -> list[list[int]]:
    fixture = tmp_path / "ghost.txt"
    _write_fixture(fixture, chains)
    env = os.environ.copy()
    env["QUBLAR_GHOST_FIXTURE"] = str(fixture)
    run = subprocess.run(
        ["cargo", "test", "-p", "lyth", "--test", "ghost_gate", "--", "--nocapture"],
        cwd=LYTH,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,
    )
    assert run.returncode == 0, run.stdout + "\n" + run.stderr
    rows = []
    for line in run.stdout.splitlines():
        if not line.startswith("w "):
            continue
        bits = [int(part, 16) for part in line.split()[1:]]
        assert len(bits) == 8, line
        rows.append(bits)
    assert rows, run.stdout
    return rows


def _f32(bits: int) -> float:
    return float(np.float32(struct.unpack("<f", struct.pack("<I", bits))[0]))


def _dense_rank4() -> np.ndarray:
    rng = np.random.default_rng(0)
    cores = [rng.standard_normal((1, 2, 4))]
    for _ in range(8):
        cores.append(rng.standard_normal((4, 2, 4)))
    cores.append(rng.standard_normal((4, 2, 1)))
    acc = cores[0]
    for core in cores[1:]:
        acc = np.tensordot(acc, core, axes=([acc.ndim - 1], [0]))
    assert acc.shape[0] == 1 and acc.shape[-1] == 1
    return acc.reshape(acc.shape[1:-1]).astype(np.float64)


def test_declined_does_not_run():
    rng = np.random.default_rng(0)
    values = rng.standard_normal(2**6) + 1j * rng.standard_normal(2**6)
    values /= np.linalg.norm(values)
    haar = values.reshape((2,) * 6).astype(np.complex128)
    answer = verdict(haar, rel_tol=REL_TOL)
    assert answer.kind is Kind.DECLINED
    assert answer.tt is None
    assert _steps(answer) is None


def test_bond_above_chi2_does_not_run():
    answer = verdict(_dense_rank4(), rel_tol=1e-8)
    assert answer.kind is Kind.COMPRESSED
    assert answer.tt is not None
    assert max(answer.tt.ranks) > 2
    assert _steps(answer) is None


@pytest.mark.ecosystem
def test_roi_marginals_match_on_both_machines(tmp_path: Path):
    assert ROI.is_file(), "missing QuBLAR build/ising_out_roi.txt"
    roi = _load(BRANCHES, "branches_blaze")
    qgpu = _load(QGPU, "qgpu_zipper_roi")
    _, _, _, h, j = roi.load_roi(ROI)
    posterior = roi.exact_posterior(h, j)
    n = len(h)
    bits = (np.arange(1 << n)[:, None] >> (n - 1 - np.arange(n))) & 1
    exact = (posterior[:, None] * bits).sum(axis=0)
    answer = verdict(posterior.reshape((2,) * n), rel_tol=REL_TOL)
    assert answer.kind is Kind.COMPRESSED
    assert answer.tt is not None
    prepared = _steps(answer)
    assert prepared is not None
    deq, packed, bra, chains = prepared
    rows = _run_machines(chains, tmp_path)

    unpacked_bra = qgpu._unpack(bra)
    finals = []
    for project in [None, *range(len(packed))]:
        transfer = np.zeros((2, 2), dtype=np.complex64)
        transfer[0, 0] = np.float32(1.0)
        for index, lanes in enumerate(packed):
            ket_lanes = _zero_phys0(lanes) if project == index else lanes
            transfer = qgpu._zipper(transfer, qgpu._unpack(ket_lanes), unpacked_bra)
        finals.append(transfer)

    assert len(rows) == len(finals)
    for row, transfer in zip(rows, finals):
        got_re = _f32(row[0])
        got_im = _f32(row[1])
        assert np.float32(got_re) == np.float32(transfer[0, 0].real)
        assert np.float32(got_im) == np.float32(transfer[0, 0].imag)
        assert row[1] == 0

    norm = _f32(rows[0][0])
    got = np.array([_f32(row[0]) / norm for row in rows[1:]])
    reference = roi.tt_marginals(deq.cores)
    assert np.abs(got - reference).max() <= 1e-6
    assert np.abs(got - exact).max() <= INT8_VS_EXACT
