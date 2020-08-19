#include <unistd.h>
#include <sys/time.h>
#include <stdio.h>

// Returns a clockticks variable
clock_t get_clockticks()
{
    clock_t t1;
    return clock();
}

// Returns the number of seconds that elapsed during the program
// Ignores other programs (or OS) that may interrupt
// May also ignore other threads of this program, if used in multithreaded (parallel) setting
double get_clock_seconds(clock_t start, clock_t end)
{
    return (end - start) / (double)CLOCKS_PER_SEC;
}

// Returns the current time of the day
double get_time()
{
    struct timeval timecheck;
    gettimeofday(&timecheck, NULL);
    return (double)timecheck.tv_sec + (double)timecheck.tv_usec*1e-6;
}

// Returns the elapsed time between the start time of the day and end time of the day
// Does not ignore OS operations
// Also measures any threads that interrupt (or idle time) in a parallel setting
double get_seconds(double start, double end)
{
    return end - start;
}


// Return the rate of data movement
// i.e. the rate at with data is read from memory
double get_rate(double seconds, long bytes)
{
    return bytes / seconds;
}

// Transform the rate to Gigabytes/second (more legible when printing)
double get_grate(double rate)
{
    return rate * 1e-9;
}
