#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>
#include <vector>

void segfault_bug()
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int n = 100;
    std::vector<int> send_buffer;
    std::vector<int> recv_buffer;
    std::vector<MPI_Request> send_req(num_procs);
    MPI_Status recv_status;
    int proc;
    int tag = 3295;
    int size;
    int ctr = 0;
    for (int i = 0; i < num_procs; i++)
    {
        size = rand() / (RAND_MAX/100);
        for (int j = 0; j < size; j++)
            send_buffer.push_back(rand());
        MPI_Isend(&(send_buffer[ctr]), size, MPI_INT, i, tag, MPI_COMM_WORLD, &(send_req[i]));
        ctr += size;
    }

    for (int i = 0; i < num_procs; i++)
    {
        MPI_Probe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &recv_status);
        proc = recv_status.MPI_SOURCE;
        MPI_Get_count(&recv_status, MPI_INT, &size);
        if (size > recv_buffer.size()) recv_buffer.resize(size);
        MPI_Recv(recv_buffer.data(), size, MPI_INT, proc, tag, MPI_COMM_WORLD, &recv_status);
    }

    MPI_Waitall(num_procs, send_req.data(), MPI_STATUSES_IGNORE);
}

void logical_bug()
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int n = 100;

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; i++)
        {

        }
    }
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    segfault_bug();

    return MPI_Finalize();
}
