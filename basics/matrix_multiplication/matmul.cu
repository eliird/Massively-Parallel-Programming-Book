__global__ matMul(int *a, int *b){

}

void init_matrices(int *a, int *b, int n){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int temp_sum = 0;

    if ((row < n) && (col < n)){
        // iterate over row and down column
        for(int k=0; k < n; k++){
            temp_sum += a[row * n + k] + b[k *n + col];
        }

        c[row *n + col] = temp_sum;
    }
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
    matrixMul<<<grid, threads>>> (d_a, d_b, d_c, n);

    //copy back to host
    cudaMemcpy(h_c, d_c, bytes, cydaMemcpyDeviceToHost);

    check_answers(h_a, h_b, h_c);

    printf("completed");

    return 0;
}