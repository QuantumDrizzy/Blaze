// Build the CUDA SVD (cuSOLVER) into a static lib and link it — only with --features cuda.
// Uses the exact nvcc invocation already verified standalone (sm_120, CUDA 13).
// Must be built from a vcvars / x64 Native Tools env so nvcc finds the MSVC host compiler.
use std::path::Path;
use std::process::Command;

fn main() {
    if std::env::var("CARGO_FEATURE_CUDA").is_err() {
        return; // CPU-only build: nothing to do.
    }

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let cu = "../../cuda/blaze_svd.cu";
    println!("cargo:rerun-if-changed={cu}");

    let lib_path = format!("{out_dir}/blaze_svd.lib");
    let status = Command::new("nvcc")
        .args(["-O2", "-arch=sm_120", "-lib", cu, "-o", &lib_path])
        .status()
        .expect("failed to run nvcc — build from an x64 Native Tools / vcvars prompt");
    assert!(status.success(), "nvcc failed to build blaze_svd.cu");

    // Link our static lib + cuSOLVER + CUDA runtime.
    println!("cargo:rustc-link-search=native={out_dir}");
    println!("cargo:rustc-link-lib=static=blaze_svd");

    let cuda_path = std::env::var("CUDA_PATH").unwrap_or_else(|_| {
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0".to_string()
    });
    let libdir = format!("{cuda_path}/lib/x64");
    if Path::new(&libdir).exists() {
        println!("cargo:rustc-link-search=native={libdir}");
    }
    println!("cargo:rustc-link-lib=cusolver");
    println!("cargo:rustc-link-lib=cudart");
}
