#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "timer.h"

// compile with gcc -o hello_world_omp hello_world_omp.c -fopenmp

// Main Method
int main(int argc, char* argv[])
{
    int n_threads, thread_id;

    // OpenMP statement, each thread has own n_threads and thread_id variables (so private)
    #pragma omp parallel private(n_threads, thread_id)
    {
        // Get number of threads and id of current thread
        n_threads = omp_get_num_threads();
        thread_id = omp_get_thread_num();

        // Print hello world
        printf("Hello World from Thread %d\n", thread_id);

        // Have thread 0 print the number of threads
        if (thread_id == 0) printf("Number of threads is %d\n", n_threads);
    }


    return 0;
}
