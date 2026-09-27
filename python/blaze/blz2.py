"""BLZ2: a verdict on disk.

BLZ1 is untouched. This file carries the certificate and the cores only when
the kind is compressed. The last 32 bytes are the SHA-256 of everything
before them. A flipped byte is rejected. Passing the original tensor
recomputes ``verdict`` and rejects a file that disagrees with it.
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .tt import TT
from .verdict import Cut, Kind, Verdict, verdict

MAGIC = b"BLZ2"
VERSION = 1
HASH_LEN = 32

_KIND_TO_BYTE = {Kind.DECLINED: 0, Kind.COMPRESSED: 1, Kind.UNDECIDED: 2}
_BYTE_TO_KIND = {value: key for key, value in _KIND_TO_BYTE.items()}

PathLike = Union[str, Path]


class BlzError(ValueError):
    pass


def _f64(value: Optional[float]) -> float:
    if value is None:
        return float("nan")
    return float(value)


def _opt_f64(value: float) -> Optional[float]:
    if not np.isfinite(value):
        return None
    return float(value)


def _i32(value: Optional[int]) -> int:
    if value is None:
        return -1
    return int(value)


def _opt_i32(value: int) -> Optional[int]:
    if value < 0:
        return None
    return int(value)


def _dtype_tag(tt: TT) -> int:
    dtype = tt.dtype()
    if dtype == np.float64:
        return 0
    if dtype == np.complex128:
        return 1
    raise BlzError(f"BLZ2 only stores f64 and complex128, got {dtype}")


def _pack_number(tag: int, value: complex) -> bytes:
    if tag == 0:
        return struct.pack("<d", float(value))
    return struct.pack("<dd", float(np.real(value)), float(np.imag(value)))


def _cores_bytes(tt: TT, tag: int) -> bytes:
    out = bytearray()
    for core in tt.cores:
        array = np.ascontiguousarray(core)
        for value in array.ravel():
            out += _pack_number(tag, value)
    return bytes(out)


def _payload(answer: Verdict) -> bytes:
    if answer.kind is Kind.COMPRESSED:
        if answer.tt is None:
            raise BlzError("compressed verdict has no cores")
        tag = _dtype_tag(answer.tt)
    else:
        if answer.tt is not None:
            raise BlzError("cores are only stored for compressed")
        tag = 0
    header = struct.pack(
        "<4sBBBB",
        MAGIC,
        VERSION,
        tag,
        len(answer.shape),
        _KIND_TO_BYTE[answer.kind],
    )
    body = struct.pack(
        "<dqQQQdddddiiiI",
        float(answer.rel_tol),
        -1 if answer.max_rank is None else int(answer.max_rank),
        int(answer.nparams),
        int(answer.dense_nbytes),
        int(answer.tt_nbytes),
        float(answer.ratio),
        float(answer.absolute_bound),
        float(answer.relative_bound),
        _f64(answer.measured_rel_error),
        _f64(answer.tail_at_cap),
        _i32(answer.cap_cut),
        _i32(answer.cap_tol_rank),
        _i32(answer.cap_rank),
        len(answer.spectrum),
    )
    cuts = bytearray()
    for cut in answer.spectrum:
        cuts += struct.pack(
            "<IIIIBxxxd",
            int(cut.index),
            int(cut.tol_rank),
            int(cut.kept),
            int(cut.spectrum_len),
            1 if cut.cap_bound else 0,
            float(cut.delta),
        )
    shape = b"".join(struct.pack("<Q", int(dim)) for dim in answer.shape)
    cores = b""
    if answer.kind is Kind.COMPRESSED and answer.tt is not None:
        ranks = answer.tt.ranks
        cores = b"".join(struct.pack("<Q", int(rank)) for rank in ranks)
        cores += _cores_bytes(answer.tt, tag)
        if sum(int(core.size) for core in answer.tt.cores) != answer.nparams:
            raise BlzError("nparams does not match the cores")
        nbytes = sum(int(core.nbytes) for core in answer.tt.cores)
        if nbytes != answer.tt_nbytes:
            raise BlzError("tt_nbytes does not match the cores")
    return header + body + bytes(cuts) + shape + cores


def write_verdict(path: PathLike, answer: Verdict) -> None:
    payload = _payload(answer)
    digest = hashlib.sha256(payload).digest()
    Path(path).write_bytes(payload + digest)


def _read_exact(buf: bytes, offset: int, fmt: str) -> tuple[tuple, int]:
    size = struct.calcsize(fmt)
    if offset + size > len(buf):
        raise BlzError("truncated BLZ2")
    return struct.unpack_from(fmt, buf, offset), offset + size


def _decode_cores(buf: bytes, offset: int, shape: tuple[int, ...], tag: int) -> tuple[TT, int]:
    ndim = len(shape)
    ranks, offset = _read_exact(buf, offset, "<" + "Q" * (ndim + 1))
    cores = []
    per = 16 if tag == 1 else 8
    for index, dim in enumerate(shape):
        left, right = int(ranks[index]), int(ranks[index + 1])
        count = left * int(dim) * right
        nbytes = count * per
        if offset + nbytes > len(buf):
            raise BlzError("truncated cores")
        raw = buf[offset:offset + nbytes]
        offset += nbytes
        if tag == 0:
            data = np.frombuffer(raw, dtype="<f8").copy()
        else:
            pairs = np.frombuffer(raw, dtype="<f8").copy()
            data = pairs[0::2] + 1j * pairs[1::2]
        cores.append(data.reshape(left, int(dim), right))
    return TT(cores=cores, shape=shape, singular_values=[]), offset


def read_verdict(path: PathLike) -> Verdict:
    blob = Path(path).read_bytes()
    if len(blob) < HASH_LEN + 4:
        raise BlzError("truncated BLZ2")
    payload, digest = blob[:-HASH_LEN], blob[-HASH_LEN:]
    if hashlib.sha256(payload).digest() != digest:
        raise BlzError("BLZ2 hash does not match the bytes")
    if payload[:4] != MAGIC:
        raise BlzError("not a BLZ2 file")
    offset = 4
    (version, tag, ndim, kind_byte), offset = _read_exact(payload, offset, "<BBBB")
    if version != VERSION:
        raise BlzError(f"unsupported BLZ2 version {version}")
    kind = _BYTE_TO_KIND.get(kind_byte)
    if kind is None:
        raise BlzError(f"unknown verdict kind {kind_byte}")
    (
        rel_tol,
        max_rank_raw,
        nparams,
        dense_nbytes,
        tt_nbytes,
        ratio,
        absolute_bound,
        relative_bound,
        measured,
        tail,
        cap_cut,
        cap_tol_rank,
        cap_rank,
        n_cuts,
    ), offset = _read_exact(payload, offset, "<dqQQQdddddiiiI")
    spectrum = []
    for _ in range(n_cuts):
        (index, tol_rank, kept, spectrum_len, cap_bound, delta), offset = _read_exact(
            payload, offset, "<IIIIBxxxd"
        )
        spectrum.append(
            Cut(
                index=int(index),
                tol_rank=int(tol_rank),
                kept=int(kept),
                spectrum_len=int(spectrum_len),
                delta=float(delta),
                cap_bound=bool(cap_bound),
            )
        )
    shape_fmt = "<" + "Q" * ndim
    shape_raw, offset = _read_exact(payload, offset, shape_fmt)
    shape = tuple(int(dim) for dim in shape_raw)
    tt: Optional[TT] = None
    if kind is Kind.COMPRESSED:
        tt, offset = _decode_cores(payload, offset, shape, tag)
        got_params = sum(int(core.size) for core in tt.cores)
        got_bytes = sum(int(core.nbytes) for core in tt.cores)
        if got_params != int(nparams) or got_bytes != int(tt_nbytes):
            raise BlzError("cores do not match the certificate")
    elif offset != len(payload):
        raise BlzError("declined or undecided file carries trailing cores")
    if offset != len(payload):
        raise BlzError("trailing bytes before the hash")
    return Verdict(
        kind=kind,
        rel_tol=float(rel_tol),
        max_rank=None if int(max_rank_raw) < 0 else int(max_rank_raw),
        shape=shape,
        nparams=int(nparams),
        dense_nbytes=int(dense_nbytes),
        tt_nbytes=int(tt_nbytes),
        ratio=float(ratio),
        absolute_bound=float(absolute_bound),
        relative_bound=float(relative_bound),
        measured_rel_error=_opt_f64(float(measured)),
        cap_cut=_opt_i32(int(cap_cut)),
        cap_tol_rank=_opt_i32(int(cap_tol_rank)),
        cap_rank=_opt_i32(int(cap_rank)),
        tail_at_cap=_opt_f64(float(tail)),
        spectrum=tuple(spectrum),
        tt=tt,
    )


def verify_verdict(path: PathLike, tensor: np.ndarray) -> Verdict:
    """Read the file and recompute the verdict on ``tensor``.

    The file is rejected when the kind or the measured error disagrees.
    """
    stored = read_verdict(path)
    fresh = verdict(
        tensor,
        rel_tol=stored.rel_tol,
        max_rank=stored.max_rank,
    )
    if fresh.kind is not stored.kind:
        raise BlzError(
            f"recomputed kind {fresh.kind.value} != stored {stored.kind.value}"
        )
    if fresh.nparams != stored.nparams or fresh.tt_nbytes != stored.tt_nbytes:
        raise BlzError("recomputed cores do not match the certificate")
    if fresh.measured_rel_error is None or stored.measured_rel_error is None:
        raise BlzError("measured error missing on one side of the check")
    if abs(fresh.measured_rel_error - stored.measured_rel_error) > 1e-9:
        raise BlzError("recomputed error does not match the certificate")
    return stored
