#include <stdlib.h>
#include <stdio.h>
#include "timer.h"
// To compile with and without vectorization (in gcc):
// gcc -o dependencies dependencies.c -O1     <--- no vectorization
// Flag to vectoize : -ftree-vectorize  
// Flag needed for vectorization of X86 processors : -msse -msse2
// Flag needed for vectorization of PowerPC platforms : -maltivec
// Other optional flags (floating point reductions) : -ffast-math -fassociative-math
//
// To see what the compiler vectorizes : -fopt-info-vec (or -fopt-info-vec-optimized)
// To see what the compiler is not able to vectorize : -fopt-info-vec-missed


// Matrix-Matrix Multiplication of Doubles (Single Pointer)
void matmat(int n, double* A, double* B, double* C, int n_iter)
{
    int i, j, k;
    int iter;
    double val;
    for (iter = 0; iter < n_iter; iter++)
    {
#pragma omp parallel for private(val, i, j, k) schedule(runtime)
        for (i = 0; i < n; i++)
        {
            for (j = 0; j < n; j++)
            {
                val = A[i*n+j];
                for (k = 0; k < n; k++)
                {
                    C[i*n+k] = val * B[j*n+k];
                }
            }
        }
    }
}


double sum(int n, double* C)
{
    int i, j;
    double s = 0;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            s += C[i*n+j];
        }
    }
    return s;
}

// This program runs matrix matrix multiplication with single pointers
// Test vectorization improvements for both doubles and floats
int main(int argc, char* argv[])
{
    int i, j;
    double start, end;
    int n_access = 1000000000;

    int n = atoi(argv[1]);
    int n_iter = n_access / (n*n*n);
    n_iter = 1;

    double* A = (double*)malloc(n*n*sizeof(double));
    double* B = (double*)malloc(n*n*sizeof(double));
    double* C = (double*)malloc(n*n*sizeof(double));

    #pragma omp parallel for schedule(runtime) private(i,j)
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            A[i*n+j] = 1.0/(i*n+j+1);
            B[i*n+j] = 1.0;
            C[i*n+j] = 0;
        }
    }

    // Warm-Up 
    matmat(n, A, B, C, n_iter);

    start = get_time();
    matmat(n, A, B, C, n_iter);
    end = get_time();
    printf("N %d, Sum %e, Time Per MatMat %e\n", n, sum(n, C), (end - start)/n_iter);

    free(A);
    free(B);
    free(C);
        
    return 0;
}
