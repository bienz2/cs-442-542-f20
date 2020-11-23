#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>
#include <cuda.h>

__global__ 
void matrixMultKernel(double* A, double* B, double* C, int n)
{
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;

    if (row < n && col < n)
    {
        float val = 0;
        for (int k = 0; k < n; k++)
            val += A[row*n+k] * B[k*n+col];
        C[row*n+col] = val;
    }
}


// Shift A 'rank_row' columns
// Shift B 'rank_col' rows
// All pairs of A and B on a single process should be multiplied
// Then, send submatrix of A to neighboring process (rowwise)
// and submatrix of B to neighboring process (columnwise)
void cannon(int n, double* A, double* B, double** C_ptr)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, 
            MPI_INFO_NULL, &node_comm);
    int ppn, node_rank;
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_size(node_comm, &ppn);

    int procs_per_gpu = ppn / num_gpus;
    int gpu = node_rank / procs_per_gpu;
    cudaSetDevice(gpu);

    int i, j;

    double* C = (double*)malloc(n*n*sizeof(double));
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            C[i*n+j] = 0;
        }
    }

    double *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, n*n*sizeof(double));
    cudaMalloc((void**)&d_B, n*n*sizeof(double));
    cudaMalloc((void**)&d_C, n*n*sizeof(double));
  
    double *A2, *A3, *B2, *B3;
    cudaMalloc((void**)&A2, n*n*sizeof(double));
    cudaMalloc((void**)&A3, n*n*sizeof(double));
    cudaMalloc((void**)&B2, n*n*sizeof(double));
    cudaMalloc((void**)&B3, n*n*sizeof(double));
    
    cudaMemcpy(d_A, A, n*n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n*n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, n*n*sizeof(double), cudaMemcpyHostToDevice);

    double* send_A = d_A;
    double* send_B = d_B;
    double* recv_A = A2;
    double* recv_B = B2;
    double* tmp;

    dim3 dimBlock(32,32);
    int grid_dim = ceil(n / 32.0);
    dim3 dimGrid(grid_dim, grid_dim);

    int sq_num_procs = sqrt(num_procs);
    int rank_row = rank / sq_num_procs;
    int rank_col = rank % sq_num_procs;

    int proc, shift;
    int proc_row, proc_col;
    int tag_a = 1234;
    int tag_b = 4321;

    MPI_Request send_req_a, send_req_b, recv_req_a, recv_req_b;

    // Cannon Shift:
    // Recv A
    shift = rank_row;
    proc_col = rank_col - shift;
    if (proc_col < 0) proc_col += sq_num_procs;
    proc = rank_row * sq_num_procs + proc_col;
    MPI_Irecv(recv_A, n*n, MPI_DOUBLE, proc, tag_a, MPI_COMM_WORLD, &recv_req_a);
    
    // Recv B
    shift = rank_col;
    proc_row = rank_row - shift;
    if (proc_row < 0) proc_row += sq_num_procs;
    proc = proc_row * sq_num_procs + rank_col;
    MPI_Irecv(recv_B, n*n, MPI_DOUBLE, proc, tag_b, MPI_COMM_WORLD, &recv_req_b);

    // Send A 
    shift = rank_row;
    proc_col = rank_col + shift;
    MPI_Isend(send_A, n*n, MPI_DOUBLE, proc, tag_a, MPI_COMM_WORLD, &send_req_a);

    // Send B
    shift = rank_col;
    proc_row = rank_row + shift;
    if (proc_row >= sq_num_procs) proc_row -= sq_num_procs;
    proc = proc_row * sq_num_procs + rank_col;
    MPI_Isend(send_B, n*n, MPI_DOUBLE, proc, tag_b, MPI_COMM_WORLD, &send_req_b);

    MPI_Wait(&send_req_a, MPI_STATUS_IGNORE);
    MPI_Wait(&recv_req_a, MPI_STATUS_IGNORE);
    MPI_Wait(&send_req_b, MPI_STATUS_IGNORE);
    MPI_Wait(&recv_req_b, MPI_STATUS_IGNORE);

    tag_a++;
    tag_b++;

    // After initial shift, can multiply pairs of matrices
    matrixMultKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, n);


    recv_A = A3;
    recv_B = B3;
    send_A = A2;
    send_B = B2;

    int n_shifts = sq_num_procs - 1;
    for (i = 0; i < n_shifts; i++)
    {
        // Recv A from neighbor
        proc_col = rank_col - 1;
        if (proc_col < 0) proc_col += sq_num_procs;
        proc = rank_row * sq_num_procs + proc_col;
        MPI_Irecv(recv_A, n*n, MPI_DOUBLE, proc, tag_a, MPI_COMM_WORLD, &recv_req_a);
        
        // Recv B from neighbor
        proc_row = rank_row - 1;
        if (proc_row < 0) proc_row += sq_num_procs;
        proc = proc_row * sq_num_procs + rank_col;
        MPI_Irecv(recv_B, n*n, MPI_DOUBLE, proc, tag_b, MPI_COMM_WORLD, &recv_req_b);

        // Send A to neighbor
        proc_col = rank_col + 1;
        if (proc_col >= sq_num_procs) proc_col -= sq_num_procs;
        proc = rank_row * sq_num_procs + proc_col;
        MPI_Isend(send_A, n*n, MPI_DOUBLE, proc, tag_a, MPI_COMM_WORLD, &send_req_a);

        // Send B to neighbor
        proc_row = rank_row + 1;
        if (proc_row >= sq_num_procs) proc_row -= sq_num_procs;
        proc = proc_row * sq_num_procs + rank_col;
        MPI_Isend(send_B, n*n, MPI_DOUBLE, proc, tag_b, MPI_COMM_WORLD, &send_req_b);  

        MPI_Wait(&send_req_a, MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req_a, MPI_STATUS_IGNORE);
        MPI_Wait(&send_req_b, MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req_b, MPI_STATUS_IGNORE);

        // After each step of communication, multiply locally recvd submatrices
        matrixMultKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, n);


        tag_a++;
        tag_b++;

        tmp = send_A;
        send_A = recv_A;
        recv_A = tmp;
        tmp = send_B;
        send_B = recv_B;
        recv_B = tmp;
    }


    free(recv_A);
    free(recv_B);
    free(send_A);
    free(send_B);

    cudaMemcpy(C, d_C, n*n*sizeof(double), cudaMemcpyDeviceToHost);

    *C_ptr = C;
}

double mat_sum(int n, double* C)
{
    double sum = 0;
    int i, j;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            sum += C[i*n+j];
        }
    }
    return sum;
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int N = atoi(argv[1]);
    int sq_num_procs = sqrt(num_procs);
    int rank_row = rank / sq_num_procs;
    int rank_col = rank % sq_num_procs;

    int n = N / sq_num_procs;
    double* A = (double*)malloc(n*n*sizeof(double));
    double* B = (double*)malloc(n*n*sizeof(double));
    double* C;

    srand(rank*time(NULL));
    int i, j;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            A[i*n+j] = ((rank_row*n)+i)*N + (rank_col*n)+j+1;
            B[i*n+j] = ((rank_row*n)+i)*N + (rank_col*n)+j+1;
        }
    }
    
    double sum_C, total_sum_C;
    double start, end;


    // Time Cannon's Method/
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    cannon(n, A, B, &C);
    end = MPI_Wtime() - start;
    sum_C = mat_sum(n, C);
    MPI_Reduce(&sum_C, &total_sum_C, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("SumC %e\n", total_sum_C);
    MPI_Reduce(&end, &start, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Elapsed Time %e\n", start);
    free(C);


    free(A);
    free(B);

    MPI_Finalize();
    return 0;
}
