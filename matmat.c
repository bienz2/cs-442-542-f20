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
void matmat_dbl(int n, double* A, double* B, double* C, int n_iter)
{
    double val;
    for (int iter = 0; iter < n_iter; iter++)
    {
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                val = A[i*n+j];
                for (int k = 0; k < n; k++)
                {
                    C[i*n+k] = val * B[j*n+k];
                }
            }
        }
    }
}

// Matrix-Matrix Multiplication of Floats (Single Pointer)
void matmat_flt(int n, float* A, float* B, float* C, int n_iter)
{
    double val;
    for (int iter = 0; iter < n_iter; iter++)
    {
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                val = A[i*n+j];
                for (int k = 0; k < n; k++)
                {
                    C[i*n+k] = val * B[j*n+k];
                }
            }
        }
    }
}


// This program runs matrix matrix multiplication with single pointers
// Test vectorization improvements for both doubles and floats
int main(int argc, char* argv[])
{

    double start, end;
    int n_access = 1000000000;

    int n_bytes = atoi(argv[1]);
    int n = atoi(argv[2]);
    int n_iter = n_access / (n*n*n);

    if (n_bytes == 4)
    {
        float* A = (float*)malloc(n*n*sizeof(float));
        float* B = (float*)malloc(n*n*sizeof(float));
        float* C = (float*)malloc(n*n*sizeof(float));

        for (int i = 0; i < n*n; i++)
        {
            A[i] = 1.0/i;
            B[i] = 1.0;
            C[i] = 0;
        }

        // Warm-Up 
        matmat_flt(n, A, B, C, n_iter);

        start = get_time();
        matmat_flt(n, A, B, C, n_iter);
        end = get_time();
        printf("N %d, Time Per MatMat %e\n", n, (end - start)/n_iter);

        free(A);
        free(B);
        free(C);
    }
    else
    {
        double* A = (double*)malloc(n*n*sizeof(double));
        double* B = (double*)malloc(n*n*sizeof(double));
        double* C = (double*)malloc(n*n*sizeof(double));

        for (int i = 0; i < n*n; i++)
        {
            A[i] = 1.0/i;
            B[i] = 1.0;
            C[i] = 0;
        }

        // Warm-Up 
        matmat_dbl(n, A, B, C, n_iter);

        start = get_time();
        matmat_dbl(n, A, B, C, n_iter);
        end = get_time();
        printf("N %d, Time Per MatMat %e\n", n, (end - start)/n_iter);

        free(A);
        free(B);
        free(C);
    }
    return 0;
}
