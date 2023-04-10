////   test.cu   ////
#include "common.hpp"

const int min_ = 0;
const int max_ = 10;

__global__ void MultiplyMatKernel(I* A, I* B, I* C, int N)
{
    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;
	int DEP = blockIdx.x*blockDim.x+threadIdx.x;

    float tmpSum = 0;

    if (ROW<N && COL<N && DEP<N) 
	{
        for (int i = 0; i < N; i++) 
		{
            tmpSum += A[ROW * N + i] * B[ROW * N + i];
        }
    }
    C[ROW * N + COL] = tmpSum;
}

void Transpose(I *A, I**At, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            // copy the value at (i,j) to (j,i) in At
            (*At)[i*N + j] = A[j*N + i];
        }
    }
}

void Print(I*A, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%d ", A[i*N + j]);
        }
        printf("\n");
    }
}

int main()
{    
    I * host_a;
    I * host_b;
	I * host_b_T;
    I * host_c;
    I * device_a;
    I * device_b;
    I * device_c;
	int kernel_len;
	int length;
    dim3 threads_per_block;
    dim3 blocks_per_grid;    
    
	kernel_len = 3;
    length = kernel_len * kernel_len * 1;
    host_a = (I *) malloc(sizeof(I) * length);
    host_b = (I *) malloc(sizeof(I) * length);
	host_b_T = (I *) malloc(sizeof(I) * length);
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
        host_a[i] = i+1;
        host_b[i] = i+1;
        host_c[i] = 0;
    }

	Transpose(host_b, &host_b_T, kernel_len);

    Print(host_a, kernel_len);
    Print(host_b_T, kernel_len);

    int dimx = kernel_len;
    int dimy = kernel_len;
    int dimz = 1;

    //int max_thread = 1024;
    threads_per_block = dim3(32, 8, 4); // because, 1204 = 32*8*4 
    blocks_per_grid = dim3((dimx + threads_per_block.x - 1) / threads_per_block.x, 
	                       (dimy + threads_per_block.y - 1) / threads_per_block.y, 
						   (dimz + threads_per_block.z - 1) / threads_per_block.z);

    print_dim3("threads_per_block", threads_per_block);
    print_dim3("blocks_per_grid", blocks_per_grid);

    CHECK_CUDA_ERROR(cudaMemcpy(device_a, host_a, sizeof(I) * length, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(device_b, host_b_T, sizeof(I) * length, cudaMemcpyHostToDevice));
    
    MultiplyMatKernel<<<blocks_per_grid, threads_per_block>>>(device_a, device_b, device_c, kernel_len);
	
    CHECK_LAST_CUDA_ERROR();	
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());	
    CHECK_CUDA_ERROR(cudaMemcpy(host_c, device_c, sizeof(I) * length, cudaMemcpyDeviceToHost));
	
	Print(host_c, kernel_len);
	
    write_output_to_file(host_a, host_b, host_c, "output.txt", length);
	
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);
	
    free(host_a);
    free(host_b);
    free(host_c);
}

