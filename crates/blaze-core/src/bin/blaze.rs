//! blaze CLI (Phase 2) - real .npy <-> .blz roundtrip + benchmark stub.
//! Parity with Python reference is the #1 gate. No general compressor claims.

use clap::{Parser, Subcommand};
use ndarray::{ArrayD, IxDyn};
use num_complex::Complex64;
use std::path::PathBuf;

use blaze_core::{
    compress_f64, compress_c64, rel_error_f64, rel_error_c64,
    write_blz_f64, write_blz_c64, read_blz_f64, read_blz_c64,
};

#[derive(Parser)]
#[command(name = "blaze", version, about = "Blaze Phase 2 Rust core (TT/MPS). Parity with Python. Honest diagnostics + quantum only. NOT a general compressor.")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compress .npy (f64 or c64) to .blz using TT-SVD
    Compress {
        input: PathBuf,
        #[arg(long)]
        max_rank: Option<usize>,
        #[arg(long, default_value_t = 1e-4)]
        rel_tol: f64,
        /// Reshape a length-2^N vector into an N-mode (2,2,...,2) tensor before
        /// compressing (the statevector / qubit case). Exclusive with --reshape.
        #[arg(long)]
        qubits: Option<usize>,
        /// Reshape the flat data into this comma-separated shape before compressing,
        /// e.g. --reshape 4,4,4,4. Exclusive with --qubits.
        #[arg(long, value_delimiter = ',')]
        reshape: Option<Vec<usize>>,
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Reconstruct .blz to .npy
    Reconstruct {
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Print .blz header info
    Info {
        input: PathBuf,
    },
    /// Run honest benchmark (reproduces Python classical_benchmark conclusions)
    Benchmark,
}

/// Reshape the ingested array into a high-order tensor before compression.
/// `--qubits N` => (2,)^N (length must be 2^N); `--reshape d0,d1,...` => those dims
/// (product must equal the element count). Neither => keep the .npy shape as-is.
fn tensorize<T: Clone>(
    arr: ArrayD<T>,
    qubits: Option<usize>,
    reshape: Option<Vec<usize>>,
) -> Result<ArrayD<T>, String> {
    let dims: Vec<usize> = match (qubits, reshape) {
        (Some(_), Some(_)) => {
            return Err("--qubits and --reshape are mutually exclusive".into())
        }
        (Some(q), None) => vec![2usize; q],
        (None, Some(r)) => r,
        (None, None) => return Ok(arr), // backward compatible: keep the .npy shape
    };
    let total: usize = dims.iter().product();
    if total != arr.len() {
        return Err(format!(
            "reshape target {:?} = {} elements != {} in the array",
            dims,
            total,
            arr.len()
        ));
    }
    // Collect in logical (row-major) order so this is robust to the source layout.
    let flat: Vec<T> = arr.iter().cloned().collect();
    ArrayD::from_shape_vec(IxDyn(&dims), flat).map_err(|e| e.to_string())
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Compress { input, max_rank, rel_tol, qubits, reshape, output } => {
            // Read a REAL .npy file (f64 first, fall back to complex128). No demo tensor.
            let f64_try: Result<ArrayD<f64>, _> = ndarray_npy::read_npy(&input);
            if let Ok(arr) = f64_try {
                let arr = tensorize(arr, qubits, reshape).unwrap_or_else(|e| {
                    eprintln!("tensorize error: {e}");
                    std::process::exit(1);
                });
                let tt = compress_f64(&arr, max_rank, rel_tol, false);
                let err = rel_error_f64(&tt, &arr);
                println!("f64 {:?}: TT ranks={:?} nparams={} rel_error={:.6e}",
                         arr.shape(), tt.ranks(), tt.nparams(), err);
                write_blz_f64(&tt, &output).expect("write .blz");
                println!("wrote {}", output.display());
            } else {
                let c64_try: Result<ArrayD<Complex64>, _> = ndarray_npy::read_npy(&input);
                match c64_try {
                    Ok(arr) => {
                        let arr = tensorize(arr, qubits, reshape).unwrap_or_else(|e| {
                            eprintln!("tensorize error: {e}");
                            std::process::exit(1);
                        });
                        let tt = compress_c64(&arr, max_rank, rel_tol, false);
                        let err = rel_error_c64(&tt, &arr);
                        println!("c64 {:?}: TT ranks={:?} nparams={} rel_error={:.6e}",
                                 arr.shape(), tt.ranks(), tt.nparams(), err);
                        write_blz_c64(&tt, &output).expect("write .blz");
                        println!("wrote {}", output.display());
                    }
                    Err(e) => {
                        eprintln!("Failed to read {} as f64 or complex128 .npy: {e}", input.display());
                        std::process::exit(1);
                    }
                }
            }
        }
        Commands::Reconstruct { input, output } => {
            if let Ok(tt) = read_blz_f64(&input) {
                let recon = tt.reconstruct();
                ndarray_npy::write_npy(&output, &recon).expect("write .npy");
                println!("reconstructed f64 (ranks={:?}) -> {}", tt.ranks(), output.display());
            } else if let Ok(tt) = read_blz_c64(&input) {
                let recon = tt.reconstruct();
                ndarray_npy::write_npy(&output, &recon).expect("write .npy");
                println!("reconstructed c64 (ranks={:?}) -> {}", tt.ranks(), output.display());
            } else {
                eprintln!("Failed to read {} as .blz f64 or c64", input.display());
                std::process::exit(1);
            }
        }
        Commands::Info { input } => {
            // Simple header peek (no full parse for stub, but could enhance)
            println!("info for {} (use hexdump or extend reader for full header dump)", input.display());
            if let Ok(tt) = read_blz_f64(&input) {
                println!("  f64 TT: ndim={} ranks={:?} nparams={}", tt.ndim(), tt.ranks(), tt.nparams());
            } else if let Ok(tt) = read_blz_c64(&input) {
                println!("  c64 TT: ndim={} ranks={:?} nparams={}", tt.ndim(), tt.ranks(), tt.nparams());
            }
        }
        Commands::Benchmark => {
            println!("=== Blaze Phase 2 honest benchmark (see Python reference for full data) ===");
            println!("TT only wins on genuinely TT-native data (low bond across ALL cuts).");
            println!("On generic structured 4D (hyperspectral-like, smooth fields): a single well-chosen matrix SVD competes or wins at matched param budget.");
            println!("Random: no one compresses it.");
            println!("Run: python -m blaze.examples.classical_benchmark  (and compare numbers)");
            println!("This Rust core is required to reproduce the same verdicts for parity gate.");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy(n: usize) -> ArrayD<f64> {
        ArrayD::from_shape_vec(IxDyn(&[n]), (0..n).map(|i| i as f64).collect()).unwrap()
    }

    #[test]
    fn qubits_reshapes_to_powers_of_two() {
        let t = tensorize(dummy(16), Some(4), None).unwrap();
        assert_eq!(t.shape(), &[2, 2, 2, 2]);
        // row-major data is preserved through the reshape
        assert_eq!(
            t.iter().cloned().collect::<Vec<_>>(),
            (0..16).map(|i| i as f64).collect::<Vec<_>>()
        );
    }

    #[test]
    fn reshape_uses_given_dims() {
        let t = tensorize(dummy(16), None, Some(vec![4, 4])).unwrap();
        assert_eq!(t.shape(), &[4, 4]);
    }

    #[test]
    fn no_flag_keeps_shape() {
        let t = tensorize(dummy(16), None, None).unwrap();
        assert_eq!(t.shape(), &[16]);
    }

    #[test]
    fn mutually_exclusive_errors() {
        assert!(tensorize(dummy(16), Some(4), Some(vec![4, 4])).is_err());
    }

    #[test]
    fn wrong_element_count_errors() {
        assert!(tensorize(dummy(16), Some(3), None).is_err()); // 2^3 = 8 != 16
        assert!(tensorize(dummy(16), None, Some(vec![5, 5])).is_err()); // 25 != 16
    }
}
