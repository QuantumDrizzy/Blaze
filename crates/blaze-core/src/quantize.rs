//! blaze-core Phase 8b — second-stage core quantization (Rust port of python/blaze/quantize.py).
//!
//! Quantizes each TT core entry to a `bits`-bit signed integer code + a shared scale
//! (per-core or per-bond). Dequantization is `code · scale`. Orthogonal to TT-SVD
//! truncation; the errors COMPOSE and are reported against the dense original.
//!
//! Parity with the Python reference is bit-exact by construction:
//!   * symmetric affine `code = round(x/scale)` clipped to ±(2^(bits-1)-1),
//!   * `round` is half-to-even (numpy default) — NOT Rust's half-away-from-zero,
//!   * scales are stored f32 (the Python stores np.float32), promoted to f64 on
//!     dequantize, so the scale rounding matches too.
//!
//! Codes are kept in-memory as i16 (covers bits 2..=16); `nbytes()` always reports
//! the REAL bit-packed size `ceil(n_codes·bits/8) + 4·n_scales`, independent of the
//! in-memory dtype — exactly like the Python.

use ndarray::Array3;
use num_complex::Complex64;

use crate::TT;

/// Scale granularity: one scale per core, or one per right-bond column.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Granularity {
    PerCore,
    PerBond,
}

/// numpy-compatible round-half-to-even (banker's rounding).
/// Rust's `f64::round` rounds half away from zero, which would break parity on
/// values landing exactly on `*.5` after scaling.
#[inline]
fn round_ties_even(x: f64) -> f64 {
    let frac = (x - x.trunc()).abs();
    if frac == 0.5 {
        let floor = x.floor();
        if (floor as i64) & 1 == 0 {
            floor
        } else {
            floor + 1.0
        }
    } else {
        x.round()
    }
}

/// Quantize one real core to (codes (rl,d,rr) i16, scales f32).
fn quantize_real_core(core: &Array3<f64>, bits: u32, gran: Granularity) -> (Array3<i16>, Vec<f32>) {
    let qmax = ((1i64 << (bits - 1)) - 1) as f64; // 127@8b, 31@6b, 7@4b, 32767@16b
    let (rl, d, rr) = (core.shape()[0], core.shape()[1], core.shape()[2]);
    let mut codes = Array3::<i16>::zeros((rl, d, rr));

    match gran {
        Granularity::PerCore => {
            let amax = core.iter().fold(0.0f64, |m, &x| m.max(x.abs()));
            let scale_f32 = if amax > 0.0 { (amax / qmax) as f32 } else { 1.0f32 };
            let sden = scale_f32 as f64;
            for ((i, j, k), &x) in core.indexed_iter() {
                codes[[i, j, k]] = round_ties_even(x / sden).clamp(-qmax, qmax) as i16;
            }
            (codes, vec![scale_f32])
        }
        Granularity::PerBond => {
            // amax per right-bond column k, over the (rl·d) rows — matches the Python
            // reshape (rl·d, rr) + max over axis 0.
            let mut amax = vec![0.0f64; rr];
            for ((_i, _j, k), &x) in core.indexed_iter() {
                amax[k] = amax[k].max(x.abs());
            }
            let scales: Vec<f32> = amax
                .iter()
                .map(|&a| if a > 0.0 { (a / qmax) as f32 } else { 1.0f32 })
                .collect();
            for ((i, j, k), &x) in core.indexed_iter() {
                let sden = scales[k] as f64;
                codes[[i, j, k]] = round_ties_even(x / sden).clamp(-qmax, qmax) as i16;
            }
            (codes, scales)
        }
    }
}

fn dequantize_real_core(codes: &Array3<i16>, scales: &[f32], gran: Granularity) -> Array3<f64> {
    let (rl, d, rr) = (codes.shape()[0], codes.shape()[1], codes.shape()[2]);
    let mut out = Array3::<f64>::zeros((rl, d, rr));
    match gran {
        Granularity::PerCore => {
            let s = scales[0] as f64;
            for ((i, j, k), &c) in codes.indexed_iter() {
                out[[i, j, k]] = c as f64 * s;
            }
        }
        Granularity::PerBond => {
            for ((i, j, k), &c) in codes.indexed_iter() {
                out[[i, j, k]] = c as f64 * (scales[k] as f64);
            }
        }
    }
    out
}

/// A TT whose core entries are stored as `bits`-bit integer codes + scales.
/// Mirror of Python `blaze.quantize.QuantizedTT`. Complex cores keep separate
/// real/imag code+scale lists.
#[derive(Clone, Debug)]
pub struct QuantizedTT {
    pub codes_re: Vec<Array3<i16>>,
    pub scales_re: Vec<Vec<f32>>,
    pub codes_im: Option<Vec<Array3<i16>>>,
    pub scales_im: Option<Vec<Vec<f32>>>,
    pub shape: Vec<usize>,
    pub bits: u32,
    pub granularity: Granularity,
}

impl QuantizedTT {
    pub fn is_complex(&self) -> bool {
        self.codes_im.is_some()
    }

    /// Rebuild a real TT (panics if this was quantized from complex data).
    pub fn dequantize_f64(&self) -> TT<f64> {
        assert!(!self.is_complex(), "dequantize_f64 on complex QuantizedTT");
        let cores = self
            .codes_re
            .iter()
            .zip(self.scales_re.iter())
            .map(|(c, s)| dequantize_real_core(c, s, self.granularity))
            .collect();
        TT { cores, shape: self.shape.clone(), singular_values: vec![] }
    }

    /// Rebuild a complex TT (panics if this was quantized from real data).
    pub fn dequantize_c64(&self) -> TT<Complex64> {
        let im_codes = self.codes_im.as_ref().expect("dequantize_c64 on real QuantizedTT");
        let im_scales = self.scales_im.as_ref().unwrap();
        let mut cores = Vec::with_capacity(self.codes_re.len());
        for i in 0..self.codes_re.len() {
            let re = dequantize_real_core(&self.codes_re[i], &self.scales_re[i], self.granularity);
            let im = dequantize_real_core(&im_codes[i], &im_scales[i], self.granularity);
            let core = Array3::from_shape_fn(re.dim(), |(a, b, c)| {
                Complex64::new(re[[a, b, c]], im[[a, b, c]])
            });
            cores.push(core);
        }
        TT { cores, shape: self.shape.clone(), singular_values: vec![] }
    }

    /// Real bit-packed storage: ceil(n_codes·bits/8) payload + 4·n_scales (f32 scales).
    pub fn nbytes(&self) -> usize {
        let n_codes: usize = self.codes_re.iter().map(|c| c.len()).sum();
        let n_scales: usize = self.scales_re.iter().map(|s| s.len()).sum();
        let (nc, ns) = if self.is_complex() {
            (n_codes * 2, n_scales * 2)
        } else {
            (n_codes, n_scales)
        };
        let code_bytes = (nc * self.bits as usize).div_ceil(8);
        code_bytes + ns * 4
    }
}

/// Quantize every core of a real TT to `bits`-bit codes.
pub fn quantize_f64(tt: &TT<f64>, bits: u32, gran: Granularity) -> QuantizedTT {
    assert!((2..=16).contains(&bits), "bits must be in [2,16]");
    let mut codes_re = Vec::with_capacity(tt.cores.len());
    let mut scales_re = Vec::with_capacity(tt.cores.len());
    for core in &tt.cores {
        let (c, s) = quantize_real_core(core, bits, gran);
        codes_re.push(c);
        scales_re.push(s);
    }
    QuantizedTT {
        codes_re,
        scales_re,
        codes_im: None,
        scales_im: None,
        shape: tt.shape.clone(),
        bits,
        granularity: gran,
    }
}

/// Quantize every core of a complex TT (separate real/imag codes).
pub fn quantize_c64(tt: &TT<Complex64>, bits: u32, gran: Granularity) -> QuantizedTT {
    assert!((2..=16).contains(&bits), "bits must be in [2,16]");
    let n = tt.cores.len();
    let (mut codes_re, mut scales_re) = (Vec::with_capacity(n), Vec::with_capacity(n));
    let (mut codes_im, mut scales_im) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for core in &tt.cores {
        let re = core.mapv(|z| z.re);
        let im = core.mapv(|z| z.im);
        let (cr, sr) = quantize_real_core(&re, bits, gran);
        let (ci, si) = quantize_real_core(&im, bits, gran);
        codes_re.push(cr);
        scales_re.push(sr);
        codes_im.push(ci);
        scales_im.push(si);
    }
    QuantizedTT {
        codes_re,
        scales_re,
        codes_im: Some(codes_im),
        scales_im: Some(scales_im),
        shape: tt.shape.clone(),
        bits,
        granularity: gran,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{compress_c64, fidelity_c64, TT};
    use ndarray::{Array3, ArrayD};
    use num_complex::Complex64;

    fn fill_c(seed: u64, n: usize) -> Vec<Complex64> {
        let mut s = seed;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f64 / (1u64 << 31) as f64) - 1.0
        };
        (0..n).map(|_| Complex64::new(next(), next())).collect()
    }

    fn tt_rel_err_c64(approx: &TT<Complex64>, exact: &TT<Complex64>) -> f64 {
        let (ra, re) = (approx.reconstruct_c64(), exact.reconstruct_c64());
        let num: f64 = ra.iter().zip(re.iter()).map(|(a, b)| (a - b).norm_sqr()).sum::<f64>().sqrt();
        let den: f64 = re.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        if den == 0.0 { 0.0 } else { num / den }
    }

    #[test]
    fn error_decreases_with_bits() {
        let n = 4usize;
        let data = ArrayD::from_shape_vec(vec![n, n, n], fill_c(7, n * n * n)).unwrap();
        let tt = compress_c64(&data, Some(6), 1e-12, false);
        let err = |b: u32| tt_rel_err_c64(&quantize_c64(&tt, b, Granularity::PerBond).dequantize_c64(), &tt);
        let (e4, e8, e12) = (err(4), err(8), err(12));
        println!("rel_err: 4b={e4:.3e} 8b={e8:.3e} 12b={e12:.3e}");
        assert!(e8 < e4, "8-bit must beat 4-bit");
        assert!(e12 < e8, "12-bit must beat 8-bit");
    }

    #[test]
    fn nbytes_smaller_than_tt() {
        let n = 5usize;
        let data = ArrayD::from_shape_vec(vec![n, n, n], fill_c(11, n * n * n)).unwrap();
        let tt = compress_c64(&data, Some(6), 1e-12, false);
        let tt_bytes: usize = tt.cores.iter().map(|c| c.len() * 16).sum(); // complex128
        let q = quantize_c64(&tt, 8, Granularity::PerBond);
        assert!(q.nbytes() < tt_bytes, "int8 index must be smaller than the c128 TT");
    }

    #[test]
    fn ghz_int8_roundtrip_fidelity() {
        let s = 1.0 / 2.0_f64.sqrt();
        let mut d = ArrayD::<Complex64>::zeros(vec![2, 2]);
        d[[0, 0]] = Complex64::new(s, 0.0);
        d[[1, 1]] = Complex64::new(s, 0.0);
        let g = compress_c64(&d, Some(2), 1e-12, false);
        let gq = quantize_c64(&g, 8, Granularity::PerBond).dequantize_c64();
        assert!(fidelity_c64(&g, &gq) > 0.999, "GHZ int8 roundtrip fidelity ~1");
    }

    #[test]
    fn round_ties_even_matches_numpy() {
        // numpy: round(0.5)=0, round(1.5)=2, round(2.5)=2, round(-0.5)=0, round(-1.5)=-2
        assert_eq!(round_ties_even(0.5), 0.0);
        assert_eq!(round_ties_even(1.5), 2.0);
        assert_eq!(round_ties_even(2.5), 2.0);
        assert_eq!(round_ties_even(-0.5), 0.0);
        assert_eq!(round_ties_even(-1.5), -2.0);
        assert_eq!(round_ties_even(0.4), 0.0);
        assert_eq!(round_ties_even(0.6), 1.0);
    }

    #[test]
    fn parity_anchor_vs_python() {
        // core (1,2,3); codes captured from python blaze.quantize._quantize_real @8b.
        let core = Array3::from_shape_vec((1, 2, 3), vec![0.5, -0.25, 1.0, 0.1, 0.8, -0.6]).unwrap();

        // per_bond: column maxima 0.5, 0.8, 1.0 -> scales a/127 (f32)
        let (codes, scales) = quantize_real_core(&core, 8, Granularity::PerBond);
        assert_eq!(
            codes.iter().copied().collect::<Vec<i16>>(),
            vec![127, -40, 127, 25, 127, -76],
            "per_bond codes must match python"
        );
        let exp_pb: Vec<f32> = [0.5f64, 0.8, 1.0].iter().map(|&a| (a / 127.0) as f32).collect();
        assert_eq!(scales, exp_pb, "per_bond scales must match python (f32)");

        // per_core: global max 1.0 -> single scale 1/127; note 0.5/(1/127)=63.5 -> 64 (even)
        let (codes_pc, scales_pc) = quantize_real_core(&core, 8, Granularity::PerCore);
        assert_eq!(
            codes_pc.iter().copied().collect::<Vec<i16>>(),
            vec![64, -32, 127, 13, 102, -76],
            "per_core codes must match python (banker's rounding)"
        );
        assert_eq!(scales_pc, vec![(1.0f64 / 127.0) as f32]);
    }
}
