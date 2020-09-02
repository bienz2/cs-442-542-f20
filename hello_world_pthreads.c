#include <stdlib.h>
#include <stdio.h>
#include "timer.h"
#include <pthread.h>

// compile with gcc -o hello_world_pthreads hello_world_pthreads.c -lpthread

// Method that each thread will execute
void* hello_world(void* arg)
{
    int* arg_ptr = (int*) arg;
    printf("Hello World from thread %d\n", *arg_ptr);
    pthread_exit(0);
    return NULL;
}


// Main Method
int main(int argc, char* argv[])
{

    // Check for command line variable (n_threads)
    if (argc == 1)
    {
        printf("Please add number of threads to execute\n");
        return 0;
    }

    // Allocate n_threads pthread_t variables, and ints to hold ids
    int n_threads = atoi(argv[1]);
    pthread_t* threads = (pthread_t*)malloc(n_threads*sizeof(pthread_t));
    int* thread_ids = (int*)malloc(n_threads*sizeof(int));


    // Create pthreads, passing thread_id for each
    for (int i = 0; i < n_threads; i++)
    {
        thread_ids[i] = i;
        pthread_create(&(threads[i]), NULL, hello_world, &(thread_ids[i]));
    }

    // Call pthread_join to wait for threads to complete
    for (int i = 0; i < n_threads; i++)
    {
        pthread_join(threads[i], NULL);
    }

    free(thread_ids);
    free(threads);

    return 0;
}
