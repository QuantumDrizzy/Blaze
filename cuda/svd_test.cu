// Toolchain de-risk: minimal cuSOLVER SVD on a KNOWN matrix.
// Verifies nvcc + CUDA 13 + sm_120 (RTX 5060 Ti) + cuSOLVER link/run, with a
// correctness check (singular values must be 2.0, 1.0) before we build anything real.
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA_ERR %d line %d\n",(int)e,__LINE__);return 1;} }while(0)

int main() {
    // A (3x2), COLUMN-MAJOR (cuSOLVER/LAPACK convention). m >= n required by gesvd.
    // col0=[1,0,0], col1=[0,2,0]  => A=[[1,0],[0,2],[0,0]] => singular values {2,1}.
    const int m = 3, n = 2;
    std::vector<double> A = {1.0, 0.0, 0.0,  0.0, 2.0, 0.0};

    cusolverDnHandle_t h;
    if (cusolverDnCreate(&h) != CUSOLVER_STATUS_SUCCESS) { printf("CUSOLVER_CREATE_FAIL\n"); return 1; }

    double *dA, *dU, *dS, *dVT, *dWork;
    int *dInfo;
    CK(cudaMalloc(&dA, sizeof(double) * m * n));
    CK(cudaMalloc(&dU, sizeof(double) * m * m));
    CK(cudaMalloc(&dS, sizeof(double) * n));
    CK(cudaMalloc(&dVT, sizeof(double) * n * n));
    CK(cudaMalloc(&dInfo, sizeof(int)));
    CK(cudaMemcpy(dA, A.data(), sizeof(double) * m * n, cudaMemcpyHostToDevice));

    int lwork = 0;
    cusolverDnDgesvd_bufferSize(h, m, n, &lwork);
    CK(cudaMalloc(&dWork, sizeof(double) * lwork));

    cusolverStatus_t st = cusolverDnDgesvd(
        h, 'A', 'A', m, n, dA, m, dS, dU, m, dVT, n, dWork, lwork, nullptr, dInfo);
    if (st != CUSOLVER_STATUS_SUCCESS) { printf("GESVD_FAIL status=%d\n", (int)st); return 1; }
    CK(cudaDeviceSynchronize());

    int info = 0;
    CK(cudaMemcpy(&info, dInfo, sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<double> S(n);
    CK(cudaMemcpy(S.data(), dS, sizeof(double) * n, cudaMemcpyDeviceToHost));

    printf("SVD_S=%.6f,%.6f (expected 2.000000,1.000000)  info=%d\n", S[0], S[1], info);
    bool ok = (S[0] > 1.999 && S[0] < 2.001) && (S[1] > 0.999 && S[1] < 1.001) && info == 0;
    printf(ok ? "TOOLCHAIN_OK\n" : "TOOLCHAIN_BAD\n");

    cudaFree(dA); cudaFree(dU); cudaFree(dS); cudaFree(dVT); cudaFree(dWork); cudaFree(dInfo);
    cusolverDnDestroy(h);
    return ok ? 0 : 1;
}
