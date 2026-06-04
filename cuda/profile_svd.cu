// Profile: where does the GPU SVD time go — transfer vs cuSOLVER compute?
// Decides whether a GPU-resident pipeline (eliminate H<->D transfers) is worth it.
// Matrix 32 x 32768 = the dominant first unfold of a 32^4 tensor.
#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cusolverDn.h>

int main() {
    const int m = 32, n = 32768;              // row-major m x n == col-major (n x m) = A^T
    const int R = n, C = m;                   // feed n x m to gesvd (R >= C)
    std::vector<double> A((size_t)m * n);
    for (size_t i = 0; i < A.size(); ++i) A[i] = std::sin(i * 0.001);

    cusolverDnHandle_t h; cusolverDnCreate(&h);
    double *dA, *dU, *dS, *dVt, *dW; int* dInfo;
    cudaMalloc(&dA, sizeof(double) * (size_t)R * C);
    cudaMalloc(&dU, sizeof(double) * (size_t)R * C);
    cudaMalloc(&dS, sizeof(double) * C);
    cudaMalloc(&dVt, sizeof(double) * (size_t)C * C);
    cudaMalloc(&dInfo, sizeof(int));
    int lwork; cusolverDnDgesvd_bufferSize(h, R, C, &lwork);
    cudaMalloc(&dW, sizeof(double) * lwork);

    // warmup (CUDA context + first cuSOLVER call)
    cudaMemcpy(dA, A.data(), sizeof(double) * A.size(), cudaMemcpyHostToDevice);
    cusolverDnDgesvd(h, 'S', 'S', R, C, dA, R, dS, dU, R, dVt, C, dW, lwork, nullptr, dInfo);
    cudaDeviceSynchronize();

    cudaEvent_t e0, e1, e2, e3;
    cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2); cudaEventCreate(&e3);

    cudaEventRecord(e0);
    cudaMemcpy(dA, A.data(), sizeof(double) * A.size(), cudaMemcpyHostToDevice);   // H->D
    cudaEventRecord(e1);
    cusolverDnDgesvd(h, 'S', 'S', R, C, dA, R, dS, dU, R, dVt, C, dW, lwork, nullptr, dInfo); // compute
    cudaEventRecord(e2);
    std::vector<double> U((size_t)R * C), S(C), Vt((size_t)C * C);
    cudaMemcpy(U.data(), dU, sizeof(double) * U.size(), cudaMemcpyDeviceToHost);    // D->H (the big U)
    cudaMemcpy(S.data(), dS, sizeof(double) * C, cudaMemcpyDeviceToHost);
    cudaMemcpy(Vt.data(), dVt, sizeof(double) * Vt.size(), cudaMemcpyDeviceToHost);
    cudaEventRecord(e3); cudaEventSynchronize(e3);

    float h2d, svd, d2h;
    cudaEventElapsedTime(&h2d, e0, e1);
    cudaEventElapsedTime(&svd, e1, e2);
    cudaEventElapsedTime(&d2h, e2, e3);
    printf("PROFILE %dx%d:  H2D=%.2fms  cuSOLVER=%.2fms  D2H=%.2fms  -> transfer=%.0f%% of total\n",
           m, n, h2d, svd, d2h, 100.0 * (h2d + d2h) / (h2d + svd + d2h));
    return 0;
}
