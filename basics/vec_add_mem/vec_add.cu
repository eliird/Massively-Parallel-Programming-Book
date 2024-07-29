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

void vector_init(int *a, int *b, int n){
    for (int i =0; i < n ; i++){
        a[i] = rand()%100;
        b[i] = rand()%100;
    }

}

void error_check(int *a, int *b, int *c, int n){
    for(int i=0; i < n; i++){
        assert(c[i] == a[i] + b[i]);
    }
}

int main(){

    // Get device ID for other CUDA calls
    int id = cudaGetDevice(&id);

    int n = 1 << 16; // 65536 elments
    size_t bytes = sizeof(int) * n;
    
    // define host pointers
    int *a, *b, *c;

    // Allocate device memory
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    // initialize vectors a and b with random values between 0 and 99
    vector_init(a, b, n);

    //Threadblock size
    int BLOCK_SIZE = 256;

    //Grid Size
    int GRID_SIZE = (int)ceil(n / BLOCK_SIZE);

    // Prefetch the data to the GPU parallely while executing the code below (to improve performance)
    cudaMemPrefetchAsync(a, bytes, id);
    cudaMemPrefetchAsync(b, bytes, id);
    //Launch Kernel on default stream with shmem
    vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(a, b, c, n);

    // waut fir akk orevuiys operation before using values
    cudaDeviceSynchronize();

    // prefetch the data from device to host async
    cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId);
    //check for the errors
    error_check(a, b, c, n);

    printf("COMPLETED SUCCESSFULY\n");

    return 0;
    
}