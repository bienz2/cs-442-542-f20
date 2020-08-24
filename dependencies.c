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


// An example loop that my compiler is able to vectorize
// GCC can detect that this loop can be rewritten without dependencies
void loop(int n, float* x, float* y, float* z, int n_iter)
{
    for (int iter = 0; iter < n_iter; iter++)
    {
        for (int i = 1; i < n; i++)
        {
            x[i] += y[i];
            z[i] += x[i];
        }
    }
}

// An example loop that my compiler is not able to vectorize
void loop_not_vec(int n, float* x, float* y, float* z, int n_iter)
{
    for (int iter = 0; iter < n_iter; iter++)
    {
        for (int i = 1; i < n; i++)
        {
            z[i] = x[i] + y[i] + z[i-1];
        }
    }
}

// An example loop that some compilers may vectorize (my compiler can) 
void loop_maybe_vec(int n, float* x, float* y, float* z, int n_iter)
{
    for (int iter = 0; iter < n_iter; iter++)
    {
        for (int i = 4; i < n; i++)
        {
            z[i] = x[i] + y[i] + z[i-4];
        }
    }
}


// If your compiler cannot vectorize previous loop,
// it may be able to optimize this unrolled one
void loop_unrolled(int n, float* x, float* y, float* z, int n_iter)
{
    for (int iter = 0; iter < n_iter; iter++)
    {
        for (int i = 4; i < n; i+=4)
        {
            z[i] = x[i] + y[i] + z[i-4];
            z[i+1] = x[i+1] + y[i+1] + z[i-3];
            z[i+2] = x[i+2] + y[i+2] + z[i-2];
            z[i+3] = x[i+3] + y[i+3] + z[i-1];
        }
    }
}


double norm(int n, float* x)
{
    double sum = 0; 
    for (int i = 0; i < n; i++)
        sum += (x[i]*x[i]);
    return sum;
}

void reset_vectors(int n, float* x, float* y, float* z)
{
    for (int i = 0; i < n; i++)
    {
        z[i] = 0;
        y[i] = 1.0;
        x[i] = 1.0 / (i+1);
    }
    z[n] = 0;
}

int main(int argc, char* argv[])
{
    double start, end;
    int n_access = 1000000000;

    int n = atoi(argv[1]);
    int n_iter = n_access / n;

    float* x = (float*)malloc(n*sizeof(float));
    float* y = (float*)malloc(n*sizeof(float));
    float* z = (float*)malloc((n+1)*sizeof(float));


    // Test vectorizable loop
    reset_vectors(n, x, y, z);
    start = get_time();
    loop(n, x, y, z, n_iter);
    end = get_time();
    printf("Loop 1 (should be vectorizable) Val %e, Time %e\n", norm(n, z), end - start);


    // Test not vectorizable loop
    reset_vectors(n, x, y, z);
    start = get_time();
    loop_not_vec(n, x, y, z, n_iter);
    end = get_time();
    printf("Loop 2 (likely not vectorizable) Val %e, Time %e\n", norm(n, z), end - start);
        

    // Test Loop that might be vectorizable (can vectorize every 4 iterations)
    reset_vectors(n, x, y, z);
    start = get_time();
    loop_maybe_vec(n, x, y, z, n_iter);
    end = get_time();
    printf("Loop 3 (maybe vectorizable) Val %e, Time %e\n", norm(n, z), end - start);


    // Test Loop that might be vectorizable (can vectorize every 4 iterations)
    reset_vectors(n, x, y, z);
    start = get_time();
    loop_unrolled(n, x, y, z, n_iter);
    end = get_time();
    printf("Loop 4 (loop 3, unrolled) Val %e, Time %e\n", norm(n, z), end - start);


    // What if we add a dependency (x[0] = z[1])
    reset_vectors(n, x, y, z);
    free(x);
    x = &(z[1]);
    start = get_time();
    loop(n, x, y, z, n_iter);
    end = get_time();
    printf("Loop 1 with aliasing: Val %e, Time %e\n", norm(n, z), end - start);



    free(y);
    free(z);

    return 0;
}
