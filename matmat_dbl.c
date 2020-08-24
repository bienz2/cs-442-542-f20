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


// Matrix-Matrix Multiplication of Doubles (Double Pointer)
// Test without the restrict variables
void matmat_dbl(int n, double** restrict A, double** restrict B, double** restrict C, int n_iter)
{
    double val;
    for (int iter = 0; iter < n_iter; iter++)
    {
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                val = A[i][j];
                for (int k = 0; k < n; k++)
                {
                    C[i][k] = val * B[j][k];
                }
            }
        }
    }
}

// Matrix-Matrix multiplication of floats (double pointer)
// Test without the restrict variables
void matmat_flt(int n, float** restrict A, float** restrict B, float** restrict C, int n_iter)
{
    double val;
    for (int iter = 0; iter < n_iter; iter++)
    {
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                val = A[i][j];
                for (int k = 0; k < n; k++)
                {
                    C[i][k] = val * B[j][k];
                }
            }
        }
    }
}

// This program runs matrix matrix multiplication with double pointers
// Test vectorization improvements for both doubles and floats
// Try with and without the restrict variables
int main(int argc, char* argv[])
{

    double start, end;
    int n_access = 1000000000;

    int n_bytes = atoi(argv[1]);
    int n = atoi(argv[2]);
    int n_iter = n_access / (n*n*n);

    if (n_bytes == 4)
    {
        float** A = (float**)malloc(n*sizeof(float*));
        float** B = (float**)malloc(n*sizeof(float*));
        float** C = (float**)malloc(n*sizeof(float*));

        for (int i = 0; i < n; i++)
        {
            A[i] = (float*)malloc(n*sizeof(float));
            B[i] = (float*)malloc(n*sizeof(float));
            C[i] = (float*)malloc(n*sizeof(float));
        }

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                A[i][j] = 1.0/i;
                B[i][j] = 1.0;
                C[i][j] = 0;
            }
        }

        // Warm-Up 
        matmat_flt(n, A, B, C, n_iter);

        start = get_time();
        matmat_flt(n, A, B, C, n_iter);
        end = get_time();
        printf("N %d, Time Per MatMat %e\n", n, (end - start)/n_iter);

        for (int i = 0; i < n; i++)
        {
            free(A[i]);
            free(B[i]);
            free(C[i]);
        }

        free(A);
        free(B);
        free(C);
    }
    else
    {
        double** A = (double**)malloc(n*sizeof(double*));
        double** B = (double**)malloc(n*sizeof(double*));
        double** C = (double**)malloc(n*sizeof(double*));

        for (int i = 0; i < n; i++)
        {
            A[i] = (double*)malloc(n*sizeof(double));
            B[i] = (double*)malloc(n*sizeof(double));
            C[i] = (double*)malloc(n*sizeof(double));
        }

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                A[i][j] = 1.0/i;
                B[i][j] = 1.0;
                C[i][j] = 0;
            }
        }

        // Warm-Up 
        matmat_dbl(n, A, B, C, n_iter);

        start = get_time();
        matmat_dbl(n, A, B, C, n_iter);
        end = get_time();
        printf("N %d, Time Per MatMat %e\n", n, (end - start)/n_iter);

        for (int i = 0; i < n; i++)
        {
            free(A[i]);
            free(B[i]);
            free(C[i]);
        }

        free(A);
        free(B);
        free(C);
    }
    return 0;
}
