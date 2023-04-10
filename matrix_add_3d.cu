////    test.cu //////////

#include "common.hpp"

const int min_ = 0;
const int max_ = 10;

__global__ void AddMatrixKernel(I *A, I *B, I *C, int N) 
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.z * blockDim.z + threadIdx.z;

    if (r < N && c < N && d < N) 
    {
        int loc_c = d * N * N + c * N + r;
        int loc_a = d * N * N + c * N + r;
        int loc_b = d * N * N + c * N + r;
        C[loc_c] = A[loc_a] + B[loc_b];
		printf("%d(%d) = %d(%d) + %d(%d)\n", C[loc_c], loc_c, A[loc_a], loc_a, B[loc_b], loc_b);
    }
}

int main()
{
    I * device_a;
    I * device_b;
    I * device_c;
    I * host_a;
    I * host_b;
    I * host_c;
	int kernel_len;
    int length;
    dim3 threads_per_block;
    dim3 blocks_per_grid;
    
    kernel_len = 3;
    length = kernel_len * kernel_len * kernel_len;
    host_a = (I *) malloc(sizeof(I) * length);
    host_b = (I *) malloc(sizeof(I) * length);
    host_c = (I *) malloc(sizeof(I) * length);
    
    if (host_a == nullptr || host_b == nullptr || host_c == nullptr)
    {
        std::cerr << "Error: Memory allocation for host arrays failed." << std::endl;
        exit(1);
    }

    CHECK_CUDA_ERROR(cudaMalloc((void**) &device_a, sizeof(I) * length));
    CHECK_CUDA_ERROR(cudaMalloc((void**) &device_b, sizeof(I) * length));
    CHECK_CUDA_ERROR(cudaMalloc((void**) &device_c, sizeof(I) * length));

    for (int i = 0; i < length ; ++i) 
    {
        host_a[i] = rand_int(min_, max_);
        host_b[i] = rand_int(min_, max_);
        host_c[i] = 0;
    }

    int dimx = kernel_len;
	int dimy = kernel_len;
	int dimz = kernel_len;

    //int max_thread = 1024;
    threads_per_block = dim3(32, 8, 4); // because, 1204 = 32*8*4 
    blocks_per_grid = dim3((dimx + threads_per_block.x - 1) / threads_per_block.x, 
	                       (dimy + threads_per_block.y - 1) / threads_per_block.y, 
						   (dimz + threads_per_block.z - 1) / threads_per_block.z);

    print_dim3("threads_per_block", threads_per_block);
    print_dim3("blocks_per_grid", blocks_per_grid);

    CHECK_CUDA_ERROR(cudaMemcpy(device_a, host_a, sizeof(I) * length, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(device_b, host_b, sizeof(I) * length, cudaMemcpyHostToDevice));
    
    AddMatrixKernel<<<blocks_per_grid, threads_per_block>>>(device_a, device_b, device_c, kernel_len);
	
    CHECK_LAST_CUDA_ERROR();
	
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());	
    CHECK_CUDA_ERROR(cudaMemcpy(host_c, device_c, sizeof(I) * length, cudaMemcpyDeviceToHost));
	
    write_output_to_file(host_a, host_b, host_c, "output.txt", length);
	
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);
	
    free(host_a);
    free(host_b);
    free(host_c);
}
