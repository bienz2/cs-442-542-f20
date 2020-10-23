#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <math.h>

void split_even_odd()
{
    int rank, num_procs;
    int new_rank, new_num_procs;
    int color = rank % 2;
    MPI_Comm new_comm;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    // What happens if key is 0 instead of rank?
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &new_comm);
    MPI_Comm_rank(new_comm, &new_rank);
    MPI_Comm_size(new_comm, &new_num_procs);
    printf("Rank %d, Color %d, New Rank %d, New Size %d\n", rank, color, new_rank, new_num_procs);
    MPI_Comm_free(&new_comm);
}

void split_by_type()
{
    int rank, num_procs;
    int new_rank, new_num_procs;
    MPI_Comm new_comm;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &new_comm);
    //MPI_Comm_split_type(MPI_COMM_WORLD, OMPI_COMM_TYPE_BOARD, rank, MPI_INFO_NULL, &new_comm);
    MPI_Comm_rank(new_comm, &new_rank);
    MPI_Comm_size(new_comm, &new_num_procs);
    if (rank == 0) printf("Orig Num Procs %d, Num Procs Per BOARD %d\n", num_procs, new_num_procs);
    printf("Rank %d, Node Rank %d\n", rank, new_rank);
    MPI_Comm_free(&new_comm);
}

void split_by_type_info()
{
    int rank, num_procs;
    int new_rank, new_num_procs;
    MPI_Info info;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  
    MPI_Comm_get_info(MPI_COMM_WORLD, &info);
    int n_keys;
    MPI_Info_get_nkeys(info, &n_keys);

    char key[MPI_MAX_INFO_KEY];
    if (rank == 0)
    {
        for (int i = 0; i < n_keys; i++)
        {
            MPI_Info_get_nthkey(info, i, key);
            printf("Key[%d] = %s\n", i, key);
        }
    }
    

    MPI_Info_free(&info);
}

void inter_comm()
{
    int key, rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    MPI_Comm intra_comm;
    MPI_Comm first_inter_comm;
    MPI_Comm second_inter_comm;  // Only group 2 needs this

    key = rank % 3;
    MPI_Comm_split(MPI_COMM_WORLD, key, rank, &intra_comm);
    
    if (key == 0)
    {
        MPI_Intercomm_create(intra_comm, 0, MPI_COMM_WORLD, 1, 1, &first_inter_comm);
    }
    else if (key == 1)
    {
        MPI_Intercomm_create(intra_comm, 0, MPI_COMM_WORLD, 0, 1, &first_inter_comm);
        MPI_Intercomm_create(intra_comm, 0, MPI_COMM_WORLD, 2, 2, &second_inter_comm);
    }
    else if (key == 2)
    {
        MPI_Intercomm_create(intra_comm, 0, MPI_COMM_WORLD, 1, 2, &first_inter_comm);
    }

    int intra_rank;
    MPI_Comm_rank(intra_comm, &intra_rank);
    printf("Rank %d, Key %d, IntraRank %d\n", rank, key, intra_rank);

    if (key == 1) MPI_Comm_free(&second_inter_comm);
    MPI_Comm_free(&first_inter_comm);
    MPI_Comm_free(&intra_comm);
}

int cart_create()
{
    int key, rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    MPI_Comm cart_comm;
    int n_dims = 2;
    int* dims = malloc(n_dims*sizeof(int));
    int* periodic = malloc(n_dims*sizeof(int));
    int reorder=1;

    dims[0] = sqrt(num_procs);
    dims[1] = num_procs / dims[0];
    periodic[0] = 1;
    periodic[1] = 0;

    MPI_Cart_create(MPI_COMM_WORLD, n_dims, dims, periodic, reorder, &cart_comm); 
    int cart_rank;
    int cart_num_procs;
    MPI_Comm_rank(cart_comm, &cart_rank);
    MPI_Comm_size(cart_comm, &cart_num_procs);    

    int* coordinates = malloc(n_dims*sizeof(int));

    MPI_Cart_coords(cart_comm, cart_rank, n_dims, coordinates);

    printf("Rank %d, Coords (%d, %d)\n", rank, coordinates[0], coordinates[1]);

    free(coordinates);

    free(dims);
    free(periodic);

    MPI_Comm_free(&cart_comm);
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    cart_create();

    return MPI_Finalize();
}
