#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

// MPI_Gatherv(void *sendbuf, int sendcount, MPI_Datatype sendtype,
//         void* recvbuf, int* recvcounts, count int* displs,
//         MPI_Datatype recvtype, int root, MPI_Comm comm)
//
// MPI_Allgatherv(void *sendbuf, int sendcount, MPI_Datatype sendtype,
//         void* recvbuf, int* recvcounts, count int* displs,
//         MPI_Datatype recvtype, MPI_Comm comm)
//
// MPI_Alltoallv(void* sendbuf, int* sendcounts, int* sdispls, 
//         MPI_Datatype sendtype, void* recvbuf, int* recvcounts,
//         int* rdispls, MPI_Datatype recvtype, MPI_Comm comm)
//


void gatherv()
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int n = rank+1;
    int start, end;
    int* init_vals = (int*)malloc(n*sizeof(int));
    int* gather_vals;
    int* recvcounts;
    int* displs;
    if (rank == 0) 
    {
        gather_vals = (int*)malloc(num_procs*num_procs*sizeof(int));
        recvcounts = (int*)malloc(num_procs*sizeof(int));
        displs = (int*)malloc((num_procs+1)*sizeof(int));
        displs[0] = 0;
        for (int i = 0; i < num_procs; i++)
        {
            recvcounts[i] = i+1;
            displs[i+1] = displs[i] + recvcounts[i];
        }
    }
    for (int i = 0; i < n; i++)
        init_vals[i] = rank;

    MPI_Gatherv(init_vals, n, MPI_INT, gather_vals, recvcounts, displs, 
            MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        for (int i = 0; i < num_procs; i++)
        {
            start = displs[i];
            end = displs[i+1];
            for (int j = start; j < end; j++)
            {
                printf("Recvd %d from rank %d\n", gather_vals[j], i);
            }
        }  
    }
    
    free(init_vals);
    if (rank == 0) 
    {
        free(gather_vals);
        free(recvcounts);
        free(displs);
    }
}


void allgatherv(int print_proc)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int n = rank+1;
    int start, end;
    int* init_vals = (int*)malloc(n*sizeof(int));
    int* gather_vals = (int*)malloc(num_procs*num_procs*sizeof(int));
    int* recvcounts = (int*)malloc(num_procs*sizeof(int));
    int* displs = (int*)malloc((num_procs+1)*sizeof(int));
    displs[0] = 0;
    for (int i = 0; i < num_procs; i++)
    {
        recvcounts[i] = i+1;
        displs[i+1] = displs[i] + recvcounts[i];
    }
    for (int i = 0; i < n; i++)
        init_vals[i] = rank;

    MPI_Allgatherv(init_vals, n, MPI_INT, gather_vals, recvcounts, displs, 
            MPI_INT, MPI_COMM_WORLD);

    if (rank == print_proc)
    {
        for (int i = 0; i < num_procs; i++)
        {
            start = displs[i];
            end = displs[i+1];
            for (int j = start; j < end; j++)
            {
                printf("Recvd %d from rank %d\n", gather_vals[j], i);
            }
        }  
    }
    
    free(init_vals);
    free(gather_vals);
    free(recvcounts);
    free(displs);
}


void alltoallv()
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int max_n = num_procs*num_procs*num_procs;
    int start, end;

    int* init_vals = (int*)malloc(max_n*sizeof(int));
    int* recv_vals = (int*)malloc(max_n*sizeof(int));

    int* send_counts = (int*)malloc(num_procs*sizeof(int));
    int* recv_counts = (int*)malloc(num_procs*sizeof(int));
    int* send_displs = (int*)malloc((num_procs+1)*sizeof(int));
    int* recv_displs = (int*)malloc((num_procs+1)*sizeof(int));
    send_displs[0] = 0;
    recv_displs[0] = 0;
    for (int i = 0; i < num_procs; i++)
    {
        send_counts[i] = rank + i + 1;
        recv_counts[i] = rank + i + 1;
        send_displs[i+1] = send_displs[i] + send_counts[i];
        recv_displs[i+1] = recv_displs[i] + recv_counts[i];
    }
    for (int i = 0; i < num_procs; i++)
    {
        start = send_displs[i];
        end = send_displs[i+1];
        for (int j = start; j < end; j++)
        {
            init_vals[j] = rank;
        }
    }

    MPI_Alltoallv(init_vals, send_counts, send_displs, MPI_INT,
            recv_vals, recv_counts, recv_displs, MPI_INT, MPI_COMM_WORLD);

    for (int i = 0; i < num_procs; i++)
    {
        start = recv_displs[i];
        end = recv_displs[i+1];
        for (int j = start; j < end; j++)
        {
            printf("Rank %d recvd %d from %d\n", rank, recv_vals[j], i);
        }
    }

    free(send_counts);
    free(recv_counts);
    free(send_displs);
    free(recv_displs);
    free(init_vals);
    free(recv_vals);
}


int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    //gatherv();

    //allgatherv(1);

    //alltoallv();

    MPI_Finalize();
    return 0;
}
