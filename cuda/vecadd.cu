#include <cuda.h>
#include "../timer.h"

__global__
void vecAddKernel(float* A, float* B, float* C, int n)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}

void vecAdd(float* h_A, float* h_B, float* h_C, int n)
{
    for (int i = 0; i < n; i++)
        h_C[i] = h_A[i] + h_B[i];
}

double sum(int n, float* h_C)
{
    double s = 0;
    for (int i = 0; i < n; i++)
        s += h_C[i];
    return s;
}

int main(int argc, char* argv[])
{
    float *h_A, *h_B, *h_C;
    double t0, tfinal;
    int n = atoi(argv[1]);

    h_A = (float*)malloc(n*sizeof(float));
    h_B = (float*)malloc(n*sizeof(float));
    h_C = (float*)malloc(n*sizeof(float));
    for (int i = 0; i < n; i++)
    {
        h_A[i] = 0.5;
        h_B[i] = 0.2;
    }

    t0 = get_time();
    vecAdd(h_A, h_B, h_C, n);
    tfinal = get_time() - t0;
    printf("VecAdd Time %e, SumC %e\n", tfinal, sum(n, h_C));


    float *d_A, *d_B, *d_C;
    
    // Allocate device memory
    cudaMalloc((void**)&d_A, n*sizeof(float));
    cudaMalloc((void**)&d_B, n*sizeof(float));
    cudaMalloc((void**)&d_C, n*sizeof(float));

    // Copy A and B to device memory
    cudaMemcpy(d_A, h_A, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n*sizeof(float), cudaMemcpyHostToDevice);

    // Device vecAdd Kernel
    t0 = get_time();
    vecAddKernel<<<ceil(n/256.0), 256>>>(d_A, d_B, d_C, n);
    tfinal = get_time() - t0;

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("Error %s\n", cudaGetErrorString(err));

    // Copy C back to host memory
    cudaMemcpy(h_C, d_C, n*sizeof(float), cudaMemcpyDeviceToHost);

    printf("VecAddKernel Time %e, SumC %e\n", tfinal, sum(n, h_C));

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);
}
