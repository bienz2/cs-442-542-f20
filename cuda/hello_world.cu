#include <stdio.h>
#include <cuda.h>

__global__ void print_kernel()
{
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main()
{
    cudaDeviceReset();
    print_kernel<<<10,10>>>();
    cudaError_t err = cudaDeviceSynchronize();
    printf("Error %s\n", cudaGetErrorString(err));
    return 0;
}
