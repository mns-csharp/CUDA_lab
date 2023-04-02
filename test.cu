#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#include <cuda.h>

typedef float t;

const int min_ = 0;
const int max_ = 10;

void init_rand()
{
	srand(time(0));
}

float rand_float(float min_, float max_)
{
    float randomFloat = (max_ - min_) * ((float)rand() / RAND_MAX) + min_;

    return randomFloat;
}

int rand_int(int min_, int max_) 
{
    return rand() % (max_ - min_ + 1) + min_;
}

__global__ void kernel_func(float *arr1, float *arr2, float *outp, int length) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < length) 
    {
        outp[tid] = arr1[tid] + arr2[tid];
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
	
	
	length = 10;
	host_a = (t *) malloc(sizeof(t) * length);
	host_b = (t *) malloc(sizeof(t) * length);
	host_c = (t *) malloc(sizeof(t) * length);
	cudaMalloc((void**)&device_a, sizeof(t) * length);
	cudaMalloc((void**)&device_b, sizeof(t) * length);
	cudaMalloc((void**)&device_c, sizeof(t) * length);

	for (int i = 0; i < length ; ++i) 
	{
        host_a[i] = rand_int(min_, max_);
        host_b[i] = rand_int(min_, max_);
        host_c[i] = 0;
    }

	threads_per_block = dim3(16, 16, 1);
	blocks_per_grid = dim3((length + threads_per_block.x - 1) / threads_per_block.x,
						(length + threads_per_block.y - 1) / threads_per_block.y,
						(length + threads_per_block.z - 1) / threads_per_block.z);
								
	
	cudaMemcpy(device_a, host_a, sizeof(t) * length, cudaMemcpyHostToDevice);
	cudaMemcpy(device_b, host_b, sizeof(t) * length, cudaMemcpyHostToDevice);
	kernel_func<<<blocks_per_grid, threads_per_block>>>(device_a, device_b, device_c, length);
	cudaDeviceSynchronize();
	cudaMemcpy(host_c, device_c, sizeof(t) * length, cudaMemcpyDeviceToHost);
	
	for (int i = 0; i < length; ++i) 
	{
        std::cout<<host_a[i]<<", ";
        std::cout<<host_b[i]<<", ";
        std::cout<<host_c[i]<<"\n";
    }
	
	cudaFree(device_a);
	cudaFree(device_b);
	cudaFree(device_c);
	free(host_a);
	free(host_b);
	free(host_c);
}