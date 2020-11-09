#include <cuda.h>
#include "../timer.h"

__global__ void matrixMultKernel(float* A, float* B, float* C, int n)
{
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    if (row < n && col < n)
    {
        float val = 0;
        for (int k = 0; k < n; k++)
            val += A[row*n+k]*B[k*n+col];
        C[row*n+col] = val;
    }
}

void matrixMult(float* A, float* B, float* C, int n)
{
    float val;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            val = 0;
            for (int k = 0; k < n; k++)
                val += A[i*n+k]*B[k*n+j];
            C[i*n+j] = val;
        }
    }
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
    int size = n*n*sizeof(float);

    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);
    for (int i = 0; i < n*n; i++)
    {
        h_A[i] = 0.5;
        h_B[i] = 0.2;
    }

    t0 = get_time();
    matrixMult(h_A, h_B, h_C, n);
    tfinal = get_time() - t0;
    printf("Matrix Mult Time %e, SumC %e\n", tfinal, sum(n*n, h_C));


    float *d_A, *d_B, *d_C;

    // Allocate device memory
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy A and B to device memory
    t0 = get_time();
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Device vecAdd Kernel
    dim3 dimBlock(32,32);
    int max_grid_dim = ceil(n/32.0);
    dim3 dimGrid(max_grid_dim, max_grid_dim);
    matrixMultKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("Error %s\n", cudaGetErrorString(err));

    // Copy C back to host memory
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    tfinal = get_time() - t0;

    printf("MatrixMultKernel Time %e, SumC %e\n", tfinal, sum(n*n, h_C));

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
