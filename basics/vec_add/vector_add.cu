#include <random>
#include <cuda_runtime.h>


__global__ void vectorAdd(int *a, int *b, int *c, int n){
    //calculate global thread ID (tid)
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x
    
    // execute the addition on each thread 
    if (tid < n){
        c[tid] = a[tid] + b[tid]
    }
}

void matrix_init(int *vec, int n){
    for (int i =0; i < n ; i++){
        vec[i] = rand()%100;
    }

}

void error_check(int *a, int *b, int *c, int n){
    for(int i=0; i < n; i++){
        assert(c[i] == a[i] + b[i]);
    }
}

int main(){

    int n = 1 << 16; // 65536 elments
    size_t bytes = sizeof(int) * n;
    
    // define host pointers
    int *h_a, *h_b, *h_c;

    // define device pointers
    int *d_a, * d_b, *d_c;

    //Allocate host memory
    h_a = (int*)malloc(bytes);
    h_b = (int*)malloc(bytes);
    h_c = (int*)malloc(bytes);

    // Allocate device memory
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // initialize vectors a and b with random values between 0 and 99
    matrix_init(h_a, n);
    matrix_init(h_b, n);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    //Threadblock size
    int NUM_THREADS = 256;

    //Grid Size
    int NUM_BLOCKS = (int)ceil(n / NUM_THREADS);

    //Launch Kernel on default stream w/o shmem
    vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, n);

    //load memory from device to host
    cudaMemcpy(h_c, d_c, cudaMemcpyDeviceToHost);

    //check for the errors
    error_check(a_c, b_c, c_d, n);

    printf("COMPLETED SUCCESSFULY\n");

    return 0;
    
}