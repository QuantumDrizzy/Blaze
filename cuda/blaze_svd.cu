// blaze_svd.cu — GPU economy SVD via cuSOLVER, C-ABI for Rust FFI (Phase 3).
//
// Contract (mirrors the CPU nalgebra path so it is a drop-in):
//   blaze_cuda_svd_f64(A, m, n, U, S, Vt)
//     A : row-major m x n  (as ndarray / Rust gives it)
//     U : row-major m x k   (k = min(m,n))   -- caller allocates
//     S : k                  (descending)
//     Vt: row-major k x n
//   returns 0 on success.
//
// The hard part — and where layout bugs hide — is that cuSOLVER is COLUMN-major and
// gesvd needs rows >= cols. We feed it a column-major matrix with rows>=cols and map
// the outputs back, handling the wide (m<n) case by working on A^T. The self-test at
// the bottom reconstructs U*diag(S)*Vt and checks it equals A for BOTH a tall and a
// wide matrix (error ~1e-15) before any of this is trusted from Rust.
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cuComplex.h>

extern "C" int blaze_cuda_svd_f64(const double* A, int m, int n, double* U, double* S, double* Vt) {
    const int k = std::min(m, n);
    const bool wide = (m < n);
    // Column-major matrix M (R x C, R >= C) fed to gesvd:
    //   tall (m>=n): M = A in column-major (transpose the row-major buffer).
    //   wide (m< n): M = A^T in column-major == the row-major A buffer as-is.
    const int R = wide ? n : m;
    const int C = wide ? m : n;            // R >= C guaranteed
    std::vector<double> M((size_t)R * C);
    if (!wide) {
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                M[(size_t)i + (size_t)j * m] = A[(size_t)i * n + j];   // row-major -> col-major
    } else {
        std::copy(A, A + (size_t)R * C, M.begin());                    // A (row-major m,n) == A^T (col-major n,m)
    }

    cusolverDnHandle_t h;
    if (cusolverDnCreate(&h) != CUSOLVER_STATUS_SUCCESS) return 10;

    double *dM = nullptr, *dU = nullptr, *dS = nullptr, *dVt = nullptr, *dW = nullptr;
    int* dInfo = nullptr;
    cudaMalloc(&dM, sizeof(double) * (size_t)R * C);
    cudaMalloc(&dU, sizeof(double) * (size_t)R * C);   // economy U: R x C (jobu='S')
    cudaMalloc(&dS, sizeof(double) * C);
    cudaMalloc(&dVt, sizeof(double) * (size_t)C * C);  // economy Vt: C x C (jobvt='S')
    cudaMalloc(&dInfo, sizeof(int));
    cudaMemcpy(dM, M.data(), sizeof(double) * (size_t)R * C, cudaMemcpyHostToDevice);

    int lwork = 0;
    cusolverDnDgesvd_bufferSize(h, R, C, &lwork);
    cudaMalloc(&dW, sizeof(double) * lwork);

    cusolverStatus_t st = cusolverDnDgesvd(h, 'S', 'S', R, C, dM, R, dS, dU, R, dVt, C,
                                           dW, lwork, nullptr, dInfo);
    cudaDeviceSynchronize();
    if (st != CUSOLVER_STATUS_SUCCESS) { cusolverDnDestroy(h); return 20 + (int)st; }

    std::vector<double> Ucm((size_t)R * C), Sv(C), Vtcm((size_t)C * C);
    cudaMemcpy(Ucm.data(), dU, sizeof(double) * (size_t)R * C, cudaMemcpyDeviceToHost);
    cudaMemcpy(Sv.data(), dS, sizeof(double) * C, cudaMemcpyDeviceToHost);
    cudaMemcpy(Vtcm.data(), dVt, sizeof(double) * (size_t)C * C, cudaMemcpyDeviceToHost);

    for (int i = 0; i < k; ++i) S[i] = Sv[i];

    if (!wide) {
        // M = A. Ucm = U_A (m x k col-major), Vtcm = Vt_A (k x n col-major), C == n == k.
        for (int i = 0; i < m; ++i)
            for (int c = 0; c < k; ++c)
                U[(size_t)i * k + c] = Ucm[(size_t)i + (size_t)c * m];
        for (int c = 0; c < k; ++c)
            for (int j = 0; j < n; ++j)
                Vt[(size_t)c * n + j] = Vtcm[(size_t)c + (size_t)j * C];
    } else {
        // M = A^T = Ucm * S * Vtcm  => A = Vtcm^T * S * Ucm^T ; k == m == C.
        // U_A(i,c) = Vtcm(c,i) [col-major C x C], Vt_A(c,j) = Ucm(j,c) [col-major R x C].
        for (int i = 0; i < m; ++i)
            for (int c = 0; c < k; ++c)
                U[(size_t)i * k + c] = Vtcm[(size_t)c + (size_t)i * C];
        for (int c = 0; c < k; ++c)
            for (int j = 0; j < n; ++j)
                Vt[(size_t)c * n + j] = Ucm[(size_t)j + (size_t)c * R];
    }

    cudaFree(dM); cudaFree(dU); cudaFree(dS); cudaFree(dVt); cudaFree(dW); cudaFree(dInfo);
    cusolverDnDestroy(h);
    return 0;
}

// Complex128 (the quantum-state path). Same layout handling as f64, but the wide
// case uses the CONJUGATE-transpose A^H (the SVD is A = U S V^H), so the mapping
// back conjugates. The self-test verifies tall + wide complex reconstructions.
extern "C" int blaze_cuda_svd_c64(const cuDoubleComplex* A, int m, int n,
                                  cuDoubleComplex* U, double* S, cuDoubleComplex* Vt) {
    const int k = std::min(m, n);
    const bool wide = (m < n);
    const int R = wide ? n : m;
    const int C = wide ? m : n;
    std::vector<cuDoubleComplex> M((size_t)R * C);
    if (!wide) {
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                M[(size_t)i + (size_t)j * m] = A[(size_t)i * n + j];      // plain transpose
    } else {
        for (int a = 0; a < n; ++a)
            for (int b = 0; b < m; ++b)
                M[(size_t)a + (size_t)b * n] = cuConj(A[(size_t)b * n + a]); // A^H
    }

    cusolverDnHandle_t h;
    if (cusolverDnCreate(&h) != CUSOLVER_STATUS_SUCCESS) return 10;

    cuDoubleComplex *dM = nullptr, *dU = nullptr, *dVt = nullptr, *dW = nullptr;
    double *dS = nullptr, *dRwork = nullptr;
    int* dInfo = nullptr;
    cudaMalloc(&dM, sizeof(cuDoubleComplex) * (size_t)R * C);
    cudaMalloc(&dU, sizeof(cuDoubleComplex) * (size_t)R * C);
    cudaMalloc(&dS, sizeof(double) * C);
    cudaMalloc(&dVt, sizeof(cuDoubleComplex) * (size_t)C * C);
    cudaMalloc(&dRwork, sizeof(double) * ((size_t)5 * C + 1));
    cudaMalloc(&dInfo, sizeof(int));
    cudaMemcpy(dM, M.data(), sizeof(cuDoubleComplex) * (size_t)R * C, cudaMemcpyHostToDevice);

    int lwork = 0;
    cusolverDnZgesvd_bufferSize(h, R, C, &lwork);
    cudaMalloc(&dW, sizeof(cuDoubleComplex) * lwork);

    cusolverStatus_t st = cusolverDnZgesvd(h, 'S', 'S', R, C, dM, R, dS, dU, R, dVt, C,
                                           dW, lwork, dRwork, dInfo);
    cudaDeviceSynchronize();
    if (st != CUSOLVER_STATUS_SUCCESS) { cusolverDnDestroy(h); return 20 + (int)st; }

    std::vector<cuDoubleComplex> Ucm((size_t)R * C), Vtcm((size_t)C * C);
    std::vector<double> Sv(C);
    cudaMemcpy(Ucm.data(), dU, sizeof(cuDoubleComplex) * (size_t)R * C, cudaMemcpyDeviceToHost);
    cudaMemcpy(Sv.data(), dS, sizeof(double) * C, cudaMemcpyDeviceToHost);
    cudaMemcpy(Vtcm.data(), dVt, sizeof(cuDoubleComplex) * (size_t)C * C, cudaMemcpyDeviceToHost);

    for (int i = 0; i < k; ++i) S[i] = Sv[i];

    if (!wide) {
        for (int i = 0; i < m; ++i)
            for (int c = 0; c < k; ++c)
                U[(size_t)i * k + c] = Ucm[(size_t)i + (size_t)c * m];
        for (int c = 0; c < k; ++c)
            for (int j = 0; j < n; ++j)
                Vt[(size_t)c * n + j] = Vtcm[(size_t)c + (size_t)j * C];
    } else {
        // A = Vtcm^H S Ucm^H  => U_A(i,c)=conj(Vtcm(c,i)), Vh_A(c,j)=conj(Ucm(j,c)).
        for (int i = 0; i < m; ++i)
            for (int c = 0; c < k; ++c)
                U[(size_t)i * k + c] = cuConj(Vtcm[(size_t)c + (size_t)i * C]);
        for (int c = 0; c < k; ++c)
            for (int j = 0; j < n; ++j)
                Vt[(size_t)c * n + j] = cuConj(Ucm[(size_t)j + (size_t)c * R]);
    }

    cudaFree(dM); cudaFree(dU); cudaFree(dS); cudaFree(dVt); cudaFree(dW); cudaFree(dRwork); cudaFree(dInfo);
    cusolverDnDestroy(h);
    return 0;
}

#ifdef BLAZE_SVD_SELFTEST
#include <cstdio>
#include <cmath>
static double check(int m, int n, const std::vector<double>& A) {
    const int k = std::min(m, n);
    std::vector<double> U((size_t)m * k), S(k), Vt((size_t)k * n);
    int rc = blaze_cuda_svd_f64(A.data(), m, n, U.data(), S.data(), Vt.data());
    if (rc != 0) { printf("  rc=%d\n", rc); return 1e9; }
    // reconstruct R = U diag(S) Vt  (all row-major) and compare to A
    double num = 0, den = 0;
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            double v = 0;
            for (int c = 0; c < k; ++c) v += U[(size_t)i*k+c] * S[c] * Vt[(size_t)c*n+j];
            double a = A[(size_t)i*n+j];
            num += (v-a)*(v-a); den += a*a;
        }
    return std::sqrt(num) / std::sqrt(den);
}
static double check_c64(int m, int n, const std::vector<cuDoubleComplex>& A) {
    const int k = std::min(m, n);
    std::vector<cuDoubleComplex> U((size_t)m * k), Vt((size_t)k * n);
    std::vector<double> S(k);
    int rc = blaze_cuda_svd_c64(A.data(), m, n, U.data(), S.data(), Vt.data());
    if (rc != 0) { printf("  c64 rc=%d\n", rc); return 1e9; }
    double num = 0, den = 0;
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            cuDoubleComplex v = make_cuDoubleComplex(0, 0);
            for (int c = 0; c < k; ++c) {
                cuDoubleComplex us = cuCmul(U[(size_t)i*k+c], make_cuDoubleComplex(S[c], 0));
                v = cuCadd(v, cuCmul(us, Vt[(size_t)c*n+j]));
            }
            cuDoubleComplex a = A[(size_t)i*n+j];
            cuDoubleComplex d = cuCsub(v, a);
            num += cuCreal(d)*cuCreal(d) + cuCimag(d)*cuCimag(d);
            den += cuCreal(a)*cuCreal(a) + cuCimag(a)*cuCimag(a);
        }
    return std::sqrt(num) / std::sqrt(den);
}
int main() {
    std::vector<double> tall = {1,2, 3,4, 5,6, 7,8};            // 4x2 row-major
    std::vector<double> wide = {1,2,3,4, 5,6,7,8};              // 2x4 row-major
    double et = check(4, 2, tall);
    double ew = check(2, 4, wide);
    printf("TALL_4x2 recon_rel_err=%.3e\n", et);
    printf("WIDE_2x4 recon_rel_err=%.3e\n", ew);
    bool ok = et < 1e-10 && ew < 1e-10;

    // complex: tall 3x2 and wide 2x3
    using cc = cuDoubleComplex;
    std::vector<cc> ct = {make_cuDoubleComplex(1,1), make_cuDoubleComplex(0,2),
                          make_cuDoubleComplex(2,-1), make_cuDoubleComplex(1,0),
                          make_cuDoubleComplex(0,3), make_cuDoubleComplex(-1,1)}; // 3x2
    std::vector<cc> cw = {make_cuDoubleComplex(1,1), make_cuDoubleComplex(0,2), make_cuDoubleComplex(2,-1),
                          make_cuDoubleComplex(1,0), make_cuDoubleComplex(0,3), make_cuDoubleComplex(-1,1)}; // 2x3
    double ect = check_c64(3, 2, ct);
    double ecw = check_c64(2, 3, cw);
    printf("C64_TALL_3x2 recon_rel_err=%.3e\n", ect);
    printf("C64_WIDE_2x3 recon_rel_err=%.3e\n", ecw);
    bool okc = ect < 1e-10 && ecw < 1e-10;

    printf((ok && okc) ? "SVD_LAYOUT_ALL_OK\n" : "SVD_LAYOUT_BAD\n");
    return (ok && okc) ? 0 : 1;
}
#endif
