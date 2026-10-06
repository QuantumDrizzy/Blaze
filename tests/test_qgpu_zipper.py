"""QGPU gate: Blaze cores, packed as ZIPPER2, run on the MTLB emulator.

GHZ is real, so Blaze's per-bond int8 (real and imag apart) lands on the
same codes the instruction packs: one scale per right bond, peak over both
parts. The emulator's E[0][0] must match the same contraction in float32.
"""

from __future__ import annotations

import os
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from blaze import compress, quantize_tt

# The ISA repository (Labare, was MTLB): $LABARE_DIR, else a Labare checkout on the Desktop.
UNIBIT = Path(os.environ.get("LABARE_DIR", Path.home() / "Desktop" / "Labare"))
LIM = 127.0


def _ghz(n: int = 4) -> np.ndarray:
    psi = np.zeros(2**n, dtype=np.complex128)
    psi[0] = psi[-1] = 2.0**-0.5
    return psi.reshape((2,) * n)


def _pad(core: np.ndarray) -> np.ndarray:
    out = np.zeros((2, 2, 2), dtype=np.complex128)
    left, phys, right = core.shape
    if phys != 2 or left > 2 or right > 2:
        raise AssertionError(f"core {core.shape} does not fit chi = 2")
    out[:left, :, :right] = core
    return out


def _lanes_from_core(core: np.ndarray) -> list[int]:
    """ZIPPER2 register: int8 codes, one f32 scale per right bond."""
    lanes = [0, 0, 0, 0]

    def set_u32(idx: int, bits: int) -> None:
        lane = (idx >> 1) & 3
        half = idx & 1
        mask = ~(0xFFFFFFFF << (half * 32)) & 0xFFFFFFFFFFFFFFFF
        lanes[lane] = (lanes[lane] & mask) | ((bits & 0xFFFFFFFF) << (half * 32))

    def set_u8(idx: int, val: int) -> None:
        lane = (idx >> 3) & 3
        byte = idx & 7
        mask = ~(0xFF << (byte * 8)) & 0xFFFFFFFFFFFFFFFF
        lanes[lane] = (lanes[lane] & mask) | ((val & 0xFF) << (byte * 8))

    for right in range(2):
        peak = 0.0
        for left in range(2):
            for phys in range(2):
                value = core[left, phys, right]
                peak = max(peak, abs(float(value.real)), abs(float(value.imag)))
        scale = peak / LIM if peak > 0.0 else 1.0
        set_u32(4 + right, struct.unpack("<I", struct.pack("<f", np.float32(scale)))[0])
        for left in range(2):
            for phys in range(2):
                value = core[left, phys, right]
                idx = 2 * ((2 * left + phys) * 2 + right)
                re = int(np.clip(np.round(value.real / scale), -LIM, LIM))
                im = int(np.clip(np.round(value.imag / scale), -LIM, LIM))
                set_u8(idx, re & 0xFF)
                set_u8(idx + 1, im & 0xFF)
    return lanes


def _i8(lanes: list[int], idx: int) -> int:
    lane = (idx >> 3) & 3
    byte = idx & 7
    raw = (lanes[lane] >> (byte * 8)) & 0xFF
    return raw - 256 if raw >= 128 else raw


def _f32(lanes: list[int], idx: int) -> np.float32:
    lane = (idx >> 1) & 3
    half = idx & 1
    bits = (lanes[lane] >> (half * 32)) & 0xFFFFFFFF
    return np.float32(struct.unpack("<f", struct.pack("<I", bits))[0])


def _unpack(lanes: list[int]) -> np.ndarray:
    out = np.zeros((2, 2, 2), dtype=np.complex64)
    for left in range(2):
        for phys in range(2):
            for right in range(2):
                idx = 2 * ((2 * left + phys) * 2 + right)
                scale = _f32(lanes, 4 + right)
                out[left, phys, right] = (
                    np.float32(_i8(lanes, idx)) * scale
                    + 1j * np.float32(_i8(lanes, idx + 1)) * scale
                )
    return out


def _zipper(transfer: np.ndarray, ket: np.ndarray, bra: np.ndarray) -> np.ndarray:
    """Same accumulation order as TensorNetworkUnit::zipper2_step, in f32."""
    acc = np.zeros((2, 2, 2), dtype=np.complex64)
    for al in range(2):
        for phys in range(2):
            for br in range(2):
                re = np.float32(0.0)
                im = np.float32(0.0)
                for bl in range(2):
                    e_re = np.float32(transfer[al, bl].real)
                    e_im = np.float32(transfer[al, bl].imag)
                    b_re = np.float32(ket[bl, phys, br].real)
                    b_im = np.float32(ket[bl, phys, br].imag)
                    re = np.float32(re + np.float32(e_re * b_re) - np.float32(e_im * b_im))
                    im = np.float32(im + np.float32(e_re * b_im) + np.float32(e_im * b_re))
                acc[al, phys, br] = re + 1j * im
    out = np.zeros((2, 2), dtype=np.complex64)
    for ar in range(2):
        for br in range(2):
            re = np.float32(0.0)
            im = np.float32(0.0)
            for al in range(2):
                for phys in range(2):
                    a_re = np.float32(bra[al, phys, ar].real)
                    a_im = np.float32(bra[al, phys, ar].imag)
                    t_re = np.float32(acc[al, phys, br].real)
                    t_im = np.float32(acc[al, phys, br].imag)
                    re = np.float32(re + np.float32(a_re * t_re) + np.float32(a_im * t_im))
                    im = np.float32(im + np.float32(a_re * t_im) - np.float32(a_im * t_re))
            out[ar, br] = re + 1j * im
    return out


def _blaze_codes_match(tt, quantized) -> None:
    for index, core in enumerate(tt.cores):
        padded = _pad(core)
        lanes = _lanes_from_core(padded)
        left, _, right = core.shape
        for ell in range(left):
            for phys in range(2):
                for rr in range(right):
                    slot = 2 * ((2 * ell + phys) * 2 + rr)
                    assert _i8(lanes, slot) == int(quantized.codes_re[index][ell, phys, rr])
                    assert quantized.codes_im is not None
                    assert _i8(lanes, slot + 1) == int(quantized.codes_im[index][ell, phys, rr])


def _program(cores: list[list[int]]) -> str:
    lines = ["        .data", "ebnd:   .dword 0x000000003F800000, 0, 0, 0", 'nl:     .asciiz "\\n"']
    for index, lanes in enumerate(cores):
        words = ", ".join(f"0x{lane:016X}" for lane in lanes)
        lines.append(f"c{index}:    .dword {words}")
    lines += [
        "        .text",
        "        .global _start",
        "_start:",
        "        la      t0, ebnd",
        "        lq      s0, 0(t0)",
    ]
    for index in range(len(cores)):
        lines += [
            f"        la      t0, c{index}",
            "        lq      s1, 0(t0)",
            "        zipper2 s0, s1, s1",
        ]
    lines += [
        "        mv      t0, s0",
        "        li      t1, 0xFFFFFFFF",
        "        and     s5, t0, t1",
        "        srli    s6, t0, 32",
        "        and     s6, s6, t1",
        "        mv      a0, s5",
        "        li      a7, 7",
        "        ecall",
        "        la      a0, nl",
        "        li      a7, 6",
        "        ecall",
        "        mv      a0, s6",
        "        li      a7, 7",
        "        ecall",
        "        halt",
    ]
    return "\n".join(lines) + "\n"


def test_ghz_cores_contract_on_the_emulator(tmp_path: Path):
    if not (UNIBIT / "Cargo.toml").is_file():
        pytest.skip(f"Labare (the Unibit emulator) not found at {UNIBIT}; set LABARE_DIR")
    tensor = _ghz(4)
    tt = compress(tensor, max_rank=2, rel_tol=1e-12)
    assert max(tt.ranks) <= 2
    quantized = quantize_tt(tt, bits=8, granularity="per_bond")
    _blaze_codes_match(tt, quantized)

    packed = [_lanes_from_core(_pad(core)) for core in tt.cores]
    unpacked = [_unpack(lanes) for lanes in packed]
    transfer = np.zeros((2, 2), dtype=np.complex64)
    transfer[0, 0] = np.float32(1.0)
    for core in unpacked:
        transfer = _zipper(transfer, core, core)
    expect_re = float(np.float32(transfer[0, 0].real))
    expect_im = float(np.float32(transfer[0, 0].imag))

    path = tmp_path / "ghz_zipper.uasm"
    path.write_text(_program(packed), encoding="ascii")
    run = subprocess.run(
        ["cargo", "run", "-q", "--", "run", str(path)],
        cwd=UNIBIT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert run.returncode == 0, run.stderr
    numbers = []
    for line in run.stdout.splitlines():
        try:
            numbers.append(float(line.strip()))
        except ValueError:
            continue
    assert len(numbers) >= 2, run.stdout
    got_re = numbers[-2]
    got_im = numbers[-1]
    assert np.float32(got_re) == np.float32(expect_re)
    assert np.float32(got_im) == np.float32(expect_im)
    assert abs(float(np.float32(got_re)) - 1.0) < 1e-3
    assert abs(float(np.float32(got_im))) < 1e-3
