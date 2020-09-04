#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"

int min_value, local_list_size;
pthread_mutex_t minimum_value_lock;


// This method tries to find minimum without any locks
// This is a race condition... you might get the wrong answer!
void* find_min_race_condition(void* list_ptr)
{
    int* list = (int*) list_ptr;
    for (int i = 0; i < local_list_size; i++)
    {
        if (list[i] < min_value)
            min_value = list[i];
    }

    pthread_exit(0);
    return NULL;
}


// This method tries to find the minimum, locking inside the for loop
// This is very expensive, but you will get the correct answer
void* find_min_expensive(void *list_ptr)
{
    int* list = (int*) list_ptr;
    int local_min_value = list[0];
    for (int i = 0; i < local_list_size; i++)
    {
        pthread_mutex_lock(&minimum_value_lock);
        if (list[i] < min_value)
            min_value = list[i];
        pthread_mutex_unlock(&minimum_value_lock);
    }

    pthread_exit(0);
    return NULL;
}


// This method finds the minimum for a local variable, and then
// locks once to write to global variable
// This method will get the correct answer
void* find_min(void *list_ptr)
{
    int* list = (int*) list_ptr;
    int local_min_value = list[0];
    for (int i = 1; i < local_list_size; i++)
    {
        if (list[i] < local_min_value)
            local_min_value = list[i];
    }

    pthread_mutex_lock(&minimum_value_lock);
    if (local_min_value < min_value)
        min_value = local_min_value;
    pthread_mutex_unlock(&minimum_value_lock);

    pthread_exit(0);
    return NULL;
}


// Assuming list_size evenly divides num_threads
int main(int argc, char *argv[]) 
{
    if (argc < 2) 
    {
        printf("Please state list_size and number of threads\n");
        return 0;
    }

    int list_size = atoi(argv[1]);
    int n_threads = atoi(argv[2]);
    local_list_size = list_size / n_threads;
    int* list = (int*)malloc(list_size*sizeof(int));

    double start, end;
    int n_outer = list_size / n_threads;
    srand(time(NULL));
    int min_val, min_pos;
    list[0] = rand();
    min_val = list[0];
    min_pos = 0;
    for (int i = 1; i < list_size; i++)
    {
        list[i] = rand();
        if (list[i] < min_val)
        {
            min_val = list[i];
            min_pos = i;
        }
    }
    printf("Actual Minimum Value %d at Position %d\n", min_val, min_pos);

    pthread_t* threads = (pthread_t*)malloc(n_threads*sizeof(pthread_t));

    min_value = list[0];
    start = get_time();
    for (int i = 0; i < n_threads; i++)
    {
        pthread_create(&(threads[i]), NULL, find_min_race_condition, &(list[i*local_list_size]));
    }
    for (int i = 0; i < n_threads; i++)
    {
        pthread_join(threads[i], NULL);
    }
    end = get_time();
    printf("Race Condition Version finds Min %d in %e seconds\n", min_value, end - start);


    pthread_mutex_init(&minimum_value_lock, NULL);

    min_value = list[0];
    start = get_time();
    for (int i = 0; i < n_threads; i++)
    {
        pthread_create(&(threads[i]), NULL, find_min_expensive, &(list[i*local_list_size]));
    }
    for (int i = 0; i < n_threads; i++)
    {
        pthread_join(threads[i], NULL);
    }
    end = get_time();
    printf("Expensive Version finds Min %d in %e seconds\n", min_value, end - start);


    min_value = list[0];
    start = get_time();
    for (int i = 0; i < n_threads; i++)
    {
        pthread_create(&(threads[i]), NULL, find_min, &(list[i*local_list_size]));
    }
    for (int i = 0; i < n_threads; i++)
    {
        pthread_join(threads[i], NULL);
    }
    end = get_time();
    printf("Best Version finds Min %d in %e seconds\n", min_value, end - start);


    pthread_mutex_destroy(&minimum_value_lock);

    free(threads);
    free(list);
    return 0;   
}
