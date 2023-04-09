#include "common.hpp"

const int min_ = 0;
const int max_ = 10;


__global__ void AddMatrixKernel(t *A, t *B, t **C, int N) 
{
	int dimx = N;
	int dimy = N;
	int dimz = N;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < N && j < N && k < N) 
	{
        int loc_c = k * dimx * dimy + j * dimx + i;
        int loc_a = j * dimx + i;
        int loc_b = i * dimy + j;
        (*C)[loc_c] = A[loc_a] + B[loc_b];
    }
}

int main()
{
    t * device_a;
    t * device_b;
    t * device_c;
    t * host_a;
    t * host_b;
    t * host_c;
    int length;
    dim3 threads_per_block;
    dim3 blocks_per_grid;
	
	
	length = 1000;
	host_a = (t *) malloc(sizeof(t) * length);
	host_b = (t *) malloc(sizeof(t) * length);
	host_c = (t *) malloc(sizeof(t) * length);
	
	if (host_a == nullptr || host_b == nullptr || host_c == nullptr)
    {
        std::cerr << "Error: Memory allocation for host arrays failed." << std::endl;
        exit(1);
    }

    CHECK_CUDA_ERROR(cudaMalloc((void**) &device_a, sizeof(t) * length));
    CHECK_CUDA_ERROR(cudaMalloc((void**) &device_b, sizeof(t) * length));
    CHECK_CUDA_ERROR(cudaMalloc((void**) &device_c, sizeof(t) * length));

	for (int i = 0; i < length ; ++i) 
	{
        host_a[i] = rand_float(min_, max_);
        host_b[i] = rand_float(min_, max_);
        host_c[i] = 0;
    }

    //int max_thread = 1024;
    int max_block = 62500;
    threads_per_block = dim3(32, 8, 4); // because, 1204 = 32*8*4 
    blocks_per_grid = dim3(max_block, max_block, max_block); 
                           // ceil((length + max_block_per_grid_per_dim-1) / 8), 
			               // ceil((length + max_block_per_grid_per_dim-1) / 4));

    print_dim3("threads_per_block", threads_per_block);
    print_dim3("blocks_per_grid", blocks_per_grid);

    CHECK_CUDA_ERROR(cudaMemcpy(device_a, host_a, sizeof(t) * length, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(device_b, host_b, sizeof(t) * length, cudaMemcpyHostToDevice));
	
    AddMatrixKernel<<<blocks_per_grid, threads_per_block>>>(device_a, device_b, &device_c, 100);
	
    CHECK_LAST_CUDA_ERROR();
	
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());	
    CHECK_CUDA_ERROR(cudaMemcpy(host_c, device_c, sizeof(t) * length, cudaMemcpyDeviceToHost));
	
    write_output_to_file(host_a, host_b, host_c, "output.txt", length);
	
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);
	
    free(host_a);
    free(host_b);
    free(host_c);
}
