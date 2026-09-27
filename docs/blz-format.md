# .blz Format Specification (Phase 2)

**Magic:** `BLZ1` (4 bytes, ASCII)
**Version:** u8 = 1 (current)
**Endianness:** little-endian (all multi-byte ints)

## Header (fixed prefix)
```
bytes 0-3:   "BLZ1"
byte  4:     version (u8 = 1)
byte  5:     dtype_tag (u8):
               0 = f64
               1 = c64 (complex<f64> = two consecutive f64 real,imag per element)
byte  6:     ndim (u8)  // 2..=255
bytes 7-14:  reserved (8 bytes, zero for v1)
```

## Shape
```
u64 ndim times: shape[i]  (little-endian u64)
```

## Ranks (bond dimensions)
There are (ndim - 1) bonds between the ndim cores.
The TT always has r[0]=1, r[ndim]=1.
We store the internal ranks r[1] .. r[ndim-1]  (ndim-1 values? Wait: full ranks list has ndim+1 entries.

Actually store full ranks as u64[ndim+1], with [0] and [last] ==1.

```
u64 (ndim + 1) times: rank[i]
```

## Cores
For i in 0..ndim:
  core_i has shape (r[i], d=shape[i], r[i+1])
  Stored row-major (r, d, r_next)  -- C order, matching ndarray default.
  Each element:
    if dtype=f64: one f64
    if c64: two f64 (real, imag)  -- contiguous, same as Python np.complex128 memory

All cores concatenated in order, no padding.

Total size after header = sum over cores (r_left * d * r_right * elem_bytes)

## Notes for implementers
- Row-major means for core with indices (rl, phys, rr) the linear index is ((rl * d + phys) * r_right + rr)
- This matches Python `core.reshape(r_left, d, r_right)` + `.ravel()` or tobytes().
- Python side (after Phase 2) must implement reader/writer for interoperability and for `backend="rust"` transparent use.
- No compression on the .blz itself (the TT is already the compressed form). Can be gzipped by user if wanted.

## Example (tiny)
For a (2,3) matrix TT with ranks [1,2,1]:
- ndim=2, dtype=0 (f64)
- shape: 2, 3
- ranks: 1,2,1
- core0: (1,2,2) = 4 f64
- core1: (2,3,1) = 6 f64

## Validation
Roundtrip with the Python reference (once .blz reader added to Python) + CLI `compress` + `reconstruct` must give rel_error matching in-memory.

This format is intentionally simple and stable for v0.2.

## BLZ2

Magic `BLZ2`, version 1. Not a revision of the bytes above: a BLZ1 reader
rejects it on the magic. The body is the verdict certificate (kind, budget,
bounds, per-cut tails, shape) and, only when the kind is `compressed`, the
cores. The last 32 bytes are the SHA-256 of everything before them. A reader
recomputes that hash and rejects the file on a mismatch. Given the original
tensor, it also recomputes `verdict` and rejects a kind or a core count that
moved. Declined and undecided store no cores. Written by `blaze.blz2`.

