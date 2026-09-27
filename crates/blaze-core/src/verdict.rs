//! Blaze verdict: compressed, declined, or undecided.
//!
//! Mirrors `python/blaze/verdict.py`. Cores are attached only when compressed.
//! BLZ1 is unchanged; BLZ2 stores this certificate.

use crate::{compress_c64, compress_f64, rel_error_c64, rel_error_f64, TT};
use ndarray::ArrayD;
use num_complex::Complex64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Declined = 0,
    Compressed = 1,
    Undecided = 2,
}

impl Kind {
    pub fn from_byte(b: u8) -> Option<Self> {
        match b {
            0 => Some(Kind::Declined),
            1 => Some(Kind::Compressed),
            2 => Some(Kind::Undecided),
            _ => None,
        }
    }

    pub fn as_byte(self) -> u8 {
        self as u8
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Cut {
    pub index: u32,
    pub tol_rank: u32,
    pub kept: u32,
    pub spectrum_len: u32,
    pub delta: f64,
    pub cap_bound: bool,
}

#[derive(Clone, Debug)]
pub enum VerdictCores {
    F64(TT<f64>),
    C64(TT<Complex64>),
}

#[derive(Clone, Debug)]
pub struct Verdict {
    pub kind: Kind,
    pub rel_tol: f64,
    pub max_rank: Option<i64>,
    pub shape: Vec<usize>,
    pub nparams: u64,
    pub dense_nbytes: u64,
    pub tt_nbytes: u64,
    pub ratio: f64,
    pub absolute_bound: f64,
    pub relative_bound: f64,
    pub measured_rel_error: Option<f64>,
    pub cap_cut: Option<i32>,
    pub cap_tol_rank: Option<i32>,
    pub cap_rank: Option<i32>,
    pub tail_at_cap: Option<f64>,
    pub spectrum: Vec<Cut>,
    pub cores: Option<VerdictCores>,
}

fn unfolding_budget(rel_tol: f64, ndim: usize) -> f64 {
    let denom = ((ndim as f64) - 1.0).sqrt().max(1.0);
    rel_tol / denom
}

/// Exact mirror of Python `truncation_rank` (searchsorted on cumulative energy).
fn truncation_rank(singular_values: &[f64], eps: f64, max_rank: Option<usize>) -> (usize, usize, bool) {
    let length = singular_values.len();
    if length == 0 {
        return (0, 0, false);
    }
    let energy: Vec<f64> = singular_values.iter().map(|s| s * s).collect();
    let total: f64 = energy.iter().sum();
    let tol_rank = if total > 0.0 && eps > 0.0 {
        let target = 1.0 - eps * eps;
        let mut cum = 0.0;
        // np.searchsorted(..., side='left'): first index with cum[i] >= target;
        // if none, insert at `length`.
        let mut index = length;
        for (i, &e) in energy.iter().enumerate() {
            cum += e / total;
            if cum >= target {
                index = i;
                break;
            }
        }
        (index + 1).min(length)
    } else {
        length
    };
    let mut kept = tol_rank;
    let mut cap_bound = false;
    if let Some(cap) = max_rank {
        cap_bound = tol_rank > cap;
        kept = kept.min(cap);
    }
    kept = kept.max(1).min(length);
    (kept, tol_rank, cap_bound)
}

fn tail(singular_values: &[f64], kept: usize) -> f64 {
    if kept >= singular_values.len() {
        return 0.0;
    }
    singular_values[kept..]
        .iter()
        .map(|s| s * s)
        .sum::<f64>()
        .sqrt()
}

fn decide_kind(
    binding: bool,
    tt_nbytes: u64,
    dense_nbytes: u64,
    measured: Option<f64>,
    rel_tol: f64,
) -> Kind {
    if binding {
        Kind::Undecided
    } else if tt_nbytes >= dense_nbytes {
        Kind::Declined
    } else if let Some(m) = measured {
        if m <= rel_tol {
            Kind::Compressed
        } else {
            Kind::Undecided
        }
    } else {
        Kind::Undecided
    }
}

fn assemble(
    kind: Kind,
    rel_tol: f64,
    max_rank: Option<usize>,
    shape: Vec<usize>,
    dense_nbytes: u64,
    nparams: usize,
    tt_nbytes: u64,
    frobenius: f64,
    singular_values: &[Vec<f64>],
    measured: Option<f64>,
    cores: Option<VerdictCores>,
) -> Verdict {
    let eps = unfolding_budget(rel_tol, shape.len());
    let mut cuts = Vec::with_capacity(singular_values.len());
    for (index, s) in singular_values.iter().enumerate() {
        let (kept, tol_rank, cap_bound) = truncation_rank(s, eps, max_rank);
        cuts.push(Cut {
            index: index as u32,
            tol_rank: tol_rank as u32,
            kept: kept as u32,
            spectrum_len: s.len() as u32,
            delta: tail(s, kept),
            cap_bound,
        });
    }

    let ratio = if tt_nbytes > 0 {
        dense_nbytes as f64 / tt_nbytes as f64
    } else {
        f64::INFINITY
    };
    let absolute_bound = if cuts.is_empty() {
        0.0
    } else {
        cuts.iter().map(|c| c.delta * c.delta).sum::<f64>().sqrt()
    };
    let relative_bound = if frobenius > 0.0 {
        absolute_bound / frobenius
    } else {
        0.0
    };

    let binding: Vec<&Cut> = cuts.iter().filter(|c| c.cap_bound).collect();
    let (cap_cut, cap_tol_rank, cap_rank, tail_at_cap) = if binding.is_empty() {
        (None, None, None, None)
    } else {
        let chosen = binding
            .iter()
            .max_by(|a, b| {
                a.delta
                    .partial_cmp(&b.delta)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| b.index.cmp(&a.index))
            })
            .unwrap();
        (
            Some(chosen.index as i32),
            Some(chosen.tol_rank as i32),
            Some(max_rank.map(|m| m as i32).unwrap_or(chosen.kept as i32)),
            Some(chosen.delta),
        )
    };

    let delivered = if kind == Kind::Compressed {
        cores
    } else {
        None
    };

    Verdict {
        kind,
        rel_tol,
        max_rank: max_rank.map(|m| m as i64),
        shape,
        nparams: nparams as u64,
        dense_nbytes,
        tt_nbytes,
        ratio,
        absolute_bound,
        relative_bound,
        measured_rel_error: measured,
        cap_cut,
        cap_tol_rank,
        cap_rank,
        tail_at_cap,
        spectrum: cuts,
        cores: delivered,
    }
}

/// INGEST → DECOMPOSE → one of three answers. Mirrors Python `verdict` for f64.
pub fn verdict_f64(tensor: &ArrayD<f64>, rel_tol: f64, max_rank: Option<usize>) -> Verdict {
    assert!(
        rel_tol.is_finite() && rel_tol >= 0.0,
        "rel_tol must be finite and >= 0"
    );
    if let Some(m) = max_rank {
        assert!(m >= 1, "max_rank must be >= 1");
    }
    assert!(tensor.ndim() >= 2, "verdict requires ndim >= 2");

    let built = compress_f64(tensor, max_rank, rel_tol, false);
    let dense_nbytes = (tensor.len() * 8) as u64;
    let tt_nbytes = (built.nparams() * 8) as u64;
    let frobenius = tensor.iter().map(|x| x * x).sum::<f64>().sqrt();
    let measured = {
        let m = rel_error_f64(&built, tensor);
        if m.is_finite() {
            Some(m)
        } else {
            None
        }
    };

    let eps = unfolding_budget(rel_tol, tensor.ndim());
    let mut has_binding = false;
    for s in &built.singular_values {
        let (_, _, cap_bound) = truncation_rank(s, eps, max_rank);
        if cap_bound {
            has_binding = true;
            break;
        }
    }
    let kind = decide_kind(has_binding, tt_nbytes, dense_nbytes, measured, rel_tol);
    let cores = Some(VerdictCores::F64(built.clone()));
    assemble(
        kind,
        rel_tol,
        max_rank,
        tensor.shape().to_vec(),
        dense_nbytes,
        built.nparams(),
        tt_nbytes,
        frobenius,
        &built.singular_values,
        measured,
        cores,
    )
}

/// Same as `verdict_f64` for complex128 tensors.
pub fn verdict_c64(tensor: &ArrayD<Complex64>, rel_tol: f64, max_rank: Option<usize>) -> Verdict {
    assert!(
        rel_tol.is_finite() && rel_tol >= 0.0,
        "rel_tol must be finite and >= 0"
    );
    if let Some(m) = max_rank {
        assert!(m >= 1, "max_rank must be >= 1");
    }
    assert!(tensor.ndim() >= 2, "verdict requires ndim >= 2");

    let built = compress_c64(tensor, max_rank, rel_tol, false);
    let dense_nbytes = (tensor.len() * 16) as u64;
    let tt_nbytes = (built.nparams() * 16) as u64;
    let frobenius = tensor.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
    let measured = {
        let m = rel_error_c64(&built, tensor);
        if m.is_finite() {
            Some(m)
        } else {
            None
        }
    };

    let eps = unfolding_budget(rel_tol, tensor.ndim());
    let mut has_binding = false;
    for s in &built.singular_values {
        let (_, _, cap_bound) = truncation_rank(s, eps, max_rank);
        if cap_bound {
            has_binding = true;
            break;
        }
    }
    let kind = decide_kind(has_binding, tt_nbytes, dense_nbytes, measured, rel_tol);
    let cores = Some(VerdictCores::C64(built.clone()));
    assemble(
        kind,
        rel_tol,
        max_rank,
        tensor.shape().to_vec(),
        dense_nbytes,
        built.nparams(),
        tt_nbytes,
        frobenius,
        &built.singular_values,
        measured,
        cores,
    )
}
