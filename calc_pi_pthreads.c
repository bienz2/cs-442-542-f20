#include <stdlib.h>
#include <stdio.h>
#include "timer.h"
#include <pthread.h>

long n_samples, local_n_samples;
void* compute_pi(void* arg)
{
    long* arg_ptr = (long*) arg;
    double rand_x, rand_y;
    long local_n_in_circle = 0;

    // passed seed as argument because time(NULL) could give same random seed
    // for multiple threads
    srand(*arg_ptr);

    for (long i = 0; i < local_n_samples; i++)
    {
       rand_x = (double)(rand()) / RAND_MAX;  // X is between 0 and 1
       rand_y = (double)(rand()) / RAND_MAX;  // Y is between 0 and 1

       // If inside circle, add to n_in_circle
       if ((rand_x*rand_x) + (rand_y*rand_y) <= 1)
           local_n_in_circle++;
    }

    *arg_ptr = local_n_in_circle;
}


// Main Method
int main(int argc, char* argv[])
{

    // Check for command line variable (n_threads)
    if (argc == 1)
    {
        printf("Please add number of samples and number of threads to execute\n");
        return 0;
    }

    // Allocate n_threads pthread_t variables, and ints to hold ids
    n_samples = atol(argv[1]);
    int n_threads = atoi(argv[2]);
    local_n_samples = n_samples / n_threads;
    pthread_t* threads = (pthread_t*)malloc(n_threads*sizeof(pthread_t));
    long* thread_args = (long*)malloc(n_threads*sizeof(long));
    long n_in_circle = 0;


    // Create pthreads, passing thread_id for each
    for (int i = 0; i < n_threads; i++)
    {
        thread_args[i] = time(NULL);
        pthread_create(&(threads[i]), NULL, compute_pi, &(thread_args[i]));
    }

    // Call pthread_join to wait for threads to complete
    for (int i = 0; i < n_threads; i++)
    {
        pthread_join(threads[i], NULL);
        n_in_circle += thread_args[i];
    }

    
    // Pi is approximately 4 * number in circle / total number in square
    double pi = 4.0*n_in_circle / n_samples;


    printf("NSamples %ld, NThreads %d, Pi Approx %e\n", n_samples, n_threads, pi);

    free(thread_args);
    free(threads);

    return 0;
}

