#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>


// deficne the static shmem for calculation convenience

#define TILE_WIDTH 16

__global__ tiledMatMul(int *a, int *b, int *c, int n, int k, int tile_size){
    __shared__ int ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ int ds_B[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = by * tile_size + ty;
    int col = bx * tile_size + tx;
    int Cvalue = 0;

    for (int t=0; t < (n/tile_size); t++){
        //collaborative loading of tiles A and B in memory
        ds_A[ty][tx] = A[row*n + t*TILE_WIDTH+tx];
        ds_B[ty][tx] = B[(t*TILE_WIDTH+tx) *k + col];

        __syncthreads();

        for (int i=0; i<TILE_WIDTH; i++){
            Cvalue += ds_A[ty][i] * ds_B[i][tx];
            __syncthreads();
        }
            
    }
    c[row * k + col] = Cvalue  
}

void init_matrix(int *a, int n){
    for(int i = 0; i < n; i++){
        for (int j = 0; j <n; j++){
            a[i * n + j] = rand() % 10;
        }
    }
}

void init_matrices(int *a, int *b, int n){
    init_matrix(a, n);
    init_matrix(b, n);
}


void check_resutls(int *a, int *b, int *c, int n){

    int *verify_c;

    for(int i=0; i<n; i++){
        for int j=0; j<n; j++{
            for j=0; k<n; k++{
                verify_c[i*n + j] += a[i*n +k] * b[k *n +j];
            }
        }
    }

    for(int i=0; i<n; i++){
        for int j=0; j<n; j++{
            assert(c[i * n + j] == verify_c[i * n + j]);
        }
    }


}

int main(){
    //Matrix of 1024 x 1024
    int n = 1 << 10;
    size_t bytes = n * sizeof(int);

    // host pointers
    int *h_a, *h_b, *h_c;

    //Allocating the host memory
    h_a = (int*)malloc(bytes);
    h_b = (int*)malloc(bytes);
    h_c = (int*)malloc(bytes);

    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);


    init_matrices(h_a, h_b, n);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    //Threads per block
    int BLOCK_SIZE = 16;
    int GRID_SIZE = (int)ceil(n /BLOCK_SIZE);

    //use dim3 objects
    dim3 grid(GRID_SIZE, GRID_SIZE);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    //Launch the kernel
    tiledMatrixMul<<<grid, threads>>> (d_a, d_b, d_c, n);

    //copy back to host
    cudaMemcpy(h_c, d_c, bytes, cydaMemcpyDeviceToHost);

    check_answers(h_a, h_b, h_c);

    // free the memory
    free(h_a);
    free(h_b);
    free(h_c);
    
    cuda_free(d_a);
    cuda_free(d_b);
    cuda_free(d_c);

    printf("completed");

    return 0;
}