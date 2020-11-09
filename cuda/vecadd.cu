#include <cuda.h>

void vecAdd(float* h_A, float* h_B, float* h_C, int n)
{
    for (int i = 0; i < n; i++)
        h_C[i] = h_A[i] + h_B[i];
}

int main(int argc, char* argv[])
{
    float *h_A, *h_B, *h_C;
    int n = atoi(argv[1]);

    cudaMalloc((void**)&h_A, n*sizeof(float));
    cudaMalloc((void**)&h_B, n*sizeof(float));
    cudaMalloc((void**)&h_C, n*sizeof(float));

    vecAdd(h_A, h_B, h_C, n);

    cudaFree(h_A);
    cudaFree(h_B);
    cudaFree(h_C);
}
