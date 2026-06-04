//! PyO3 bindings for blaze-core (Phase 2).
//!
//! Exposes the validated Rust TT-SVD core to Python (module `blaze_py`), so
//! experiments / Cirq can drive the Rust backend. Flat-buffer marshalling
//! (Vec<T> <-> Python list) — no rust-numpy dependency needed.

use ndarray::ArrayD;
use num_complex::Complex64;
use pyo3::prelude::*;

use blaze_core::{compress_c64, compress_f64, rel_error_c64, rel_error_f64};

/// Compress a flat f64 buffer (C-order) + shape with TT-SVD.
/// Returns (reconstruction flat C-order, shape, bond ranks, relative Frobenius error).
#[pyfunction]
#[pyo3(signature = (data, shape, max_rank=None, rel_tol=1e-4))]
fn compress_f64_py(
    data: Vec<f64>,
    shape: Vec<usize>,
    max_rank: Option<usize>,
    rel_tol: f64,
) -> PyResult<(Vec<f64>, Vec<usize>, Vec<usize>, f64)> {
    let arr = ArrayD::<f64>::from_shape_vec(shape.clone(), data)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let tt = compress_f64(&arr, max_rank, rel_tol, false);
    let err = rel_error_f64(&tt, &arr);
    let recon: Vec<f64> = tt.reconstruct().iter().copied().collect(); // logical C-order
    Ok((recon, shape, tt.ranks(), err))
}

/// Same, for complex128 (the quantum-state path).
#[pyfunction]
#[pyo3(signature = (data, shape, max_rank=None, rel_tol=1e-4))]
fn compress_c64_py(
    data: Vec<Complex64>,
    shape: Vec<usize>,
    max_rank: Option<usize>,
    rel_tol: f64,
) -> PyResult<(Vec<Complex64>, Vec<usize>, Vec<usize>, f64)> {
    let arr = ArrayD::<Complex64>::from_shape_vec(shape.clone(), data)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let tt = compress_c64(&arr, max_rank, rel_tol, false);
    let err = rel_error_c64(&tt, &arr);
    let recon: Vec<Complex64> = tt.reconstruct().iter().copied().collect();
    Ok((recon, shape, tt.ranks(), err))
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compress_f64_py, m)?)?;
    m.add_function(wrap_pyfunction!(compress_c64_py, m)?)?;
    Ok(())
}
