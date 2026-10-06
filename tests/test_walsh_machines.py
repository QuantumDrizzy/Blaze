"""Hadamard on TT cores, through hadamard.lyth, on both machines.

The float64 Walsh bar stays 1e-12 in test_walsh_cores.py. This file does not
move it. The machine kernel is f32, and on the 6-bit fixture that spectrum
misses 1e-12. What is checked here is bit-exact agreement of host, PTX and
the emulator with the f32 formula, and unchanged core shapes.

The ROI uses the verdict budget already declared for that tensor, rel_tol
1e-6. A verdict that is not compressed does not run.
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

from blaze import Kind, compress, verdict

# The sibling checkouts (QuBLAR, LYTH): $BLAZE_ECOSYSTEM, else the folder that holds them here.
DESKTOP = Path(os.environ.get("BLAZE_ECOSYSTEM", Path(__file__).resolve().parents[4]))
ROI = DESKTOP / "PR0JECTS" / "RESEARCH" / "QuBLAR" / "build" / "ising_out_roi.txt"
LYTH = DESKTOP / "LYTH"
BRANCHES = DESKTOP / "PR0JECTS" / "RESEARCH" / "QuBLAR" / "tools" / "branches_blaze.py"
S64 = np.float64(2.0**-0.5)
REL_TOL = 1e-6
LANE = 8


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _bits(value: float) -> int:
    return struct.unpack("<I", np.float32(value).tobytes())[0]


def _f32(bits: int) -> np.float32:
    return np.float32(struct.unpack("<f", struct.pack("<I", bits))[0])


def _walsh_dense(p: np.ndarray) -> np.ndarray:
    out = np.asarray(p, dtype=np.float64)
    for axis in range(out.ndim):
        moved = np.moveaxis(out, axis, 0)
        lo, hi = moved[0], moved[1]
        mixed = np.stack((S64 * (lo + hi), S64 * (lo - hi)), axis=0)
        out = np.moveaxis(mixed, 0, axis)
    return out


def _hadamard_core(core: np.ndarray) -> np.ndarray:
    lo, hi = core[:, 0, :], core[:, 1, :]
    out = np.empty_like(core)
    out[:, 0, :] = S64 * (lo + hi)
    out[:, 1, :] = S64 * (lo - hi)
    return out


def _pack(cores: list[np.ndarray]) -> tuple[int, list[list[int]], list[int]]:
    planes = [[] for _ in range(4)]
    widths = []
    for core in cores:
        if core.shape[1] != 2:
            raise AssertionError(f"physical axis is {core.shape[1]}, the kernel wants 2")
        width = int(core.shape[0] * core.shape[2])
        if width > LANE:
            raise AssertionError(f"a slice of {width} does not fit one register")
        widths.append(width)
        real = [
            np.ascontiguousarray(core[:, 0, :]).ravel(),
            np.ascontiguousarray(core[:, 1, :]).ravel(),
        ]
        for phys in range(2):
            words = [_bits(value) for value in real[phys]]
            words.extend([0] * (LANE - width))
            planes[phys * 2].extend(words)
            planes[phys * 2 + 1].extend([0] * LANE)
    n = LANE * len(cores)
    assert all(len(plane) == n for plane in planes)
    return n, planes, widths


def _write(path: Path, n: int, planes: list[list[int]]) -> None:
    lines = [str(n)]
    for plane in planes:
        lines.append(" ".join(f"{word:08x}" for word in plane))
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _run(path: Path) -> dict[str, list[int]]:
    env = os.environ.copy()
    env["WALSH_FIXTURE"] = str(path)
    run = subprocess.run(
        ["cargo", "test", "-p", "lyth", "--test", "walsh_gate", "--", "--nocapture"],
        cwd=LYTH,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,
    )
    assert run.returncode == 0, run.stdout + "\n" + run.stderr
    got: dict[str, list[int]] = {}
    for line in run.stdout.splitlines():
        if not line.startswith("w "):
            continue
        name, *words = line.split()[1:]
        got[name] = [int(word, 16) for word in words]
    for name in ("q0r", "q0i", "q1r", "q1i"):
        assert name in got, run.stdout
    return got


def _rel(tt, cores, dense) -> float:
    saved = tt.cores
    tt.cores = cores
    try:
        recon = tt.reconstruct()
    finally:
        tt.cores = saved
    return float(
        np.linalg.norm((recon - dense).ravel()) / np.linalg.norm(dense.ravel())
    )


def _machine_cores(cores, got, widths):
    built = []
    s = np.float32(2.0**-0.5)
    for index, core in enumerate(cores):
        lo = index * LANE
        hi = lo + LANE
        width = widths[index]
        q0 = np.array([_f32(bits) for bits in got["q0r"][lo:hi]], dtype=np.float32)
        q1 = np.array([_f32(bits) for bits in got["q1r"][lo:hi]], dtype=np.float32)
        p0 = np.ascontiguousarray(core[:, 0, :]).ravel().astype(np.float32)
        p1 = np.ascontiguousarray(core[:, 1, :]).ravel().astype(np.float32)
        pad0 = np.zeros(LANE, dtype=np.float32)
        pad1 = np.zeros(LANE, dtype=np.float32)
        pad0[:width] = p0
        pad1[:width] = p1
        expect0 = s * (pad0 + pad1)
        expect1 = s * (pad0 - pad1)
        assert q0.tobytes() == np.asarray(expect0, dtype=np.float32).tobytes()
        assert q1.tobytes() == np.asarray(expect1, dtype=np.float32).tobytes()
        assert got["q0i"][lo:hi] == [0] * LANE
        assert got["q1i"][lo:hi] == [0] * LANE
        out = np.zeros(core.shape, dtype=np.float64)
        out[:, 0, :] = q0[:width].reshape(core.shape[0], core.shape[2])
        out[:, 1, :] = q1[:width].reshape(core.shape[0], core.shape[2])
        built.append(out)
    return built


@pytest.mark.ecosystem
def test_correlated_bits_stay_at_1e_12_in_float64_and_the_machines_match_f32(tmp_path: Path):
    p = np.zeros((2,) * 6, dtype=np.float64)
    p[(0, 0, 0, 0, 0, 0)] = 0.5
    p[(0, 0, 1, 1, 0, 0)] = 0.5
    tt = compress(p, rel_tol=1e-12)
    dense = _walsh_dense(p)
    floated = [_hadamard_core(core) for core in tt.cores]
    assert [core.shape for core in floated] == [core.shape for core in tt.cores]
    assert _rel(tt, floated, dense) <= 1e-12

    n, planes, widths = _pack(tt.cores)
    fixture = tmp_path / "walsh.txt"
    _write(fixture, n, planes)
    got = _run(fixture)
    machine = _machine_cores(tt.cores, got, widths)
    assert [core.shape for core in machine] == [core.shape for core in tt.cores]
    # The f32 spectrum on this fixture is not the 1e-12 bar. That one is the float64 check above.


@pytest.mark.ecosystem
def test_roi_hadamard_stays_inside_the_verdict_budget(tmp_path: Path):
    assert ROI.is_file(), "missing QuBLAR build/ising_out_roi.txt"
    roi = _load(BRANCHES, "branches_blaze_walsh")
    _, _, _, h, j = roi.load_roi(ROI)
    posterior = roi.exact_posterior(h, j)
    nbits = len(h)
    tensor = posterior.reshape((2,) * nbits)
    answer = verdict(tensor, rel_tol=REL_TOL)
    assert answer.kind is Kind.COMPRESSED
    assert answer.tt is not None
    tt = answer.tt
    dense = _walsh_dense(tensor)
    n, planes, widths = _pack(tt.cores)
    fixture = tmp_path / "walsh_roi.txt"
    _write(fixture, n, planes)
    machine = _machine_cores(tt.cores, _run(fixture), widths)
    assert [core.shape for core in machine] == [core.shape for core in tt.cores]
    assert _rel(tt, machine, dense) <= REL_TOL
