#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>
#include <cuda.h>
#include <fstream>
#include <iomanip>
#include <string>
#include <chrono>

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

void write_output_to_file(t* host_a, t* host_b, t* host_c, std::string fileName, int length) 
{
	std::ofstream outputFile;
	outputFile.open(fileName);

	if(outputFile.is_open()) 
	{
		for(int i = 0; i < length; i++) 
		{
			outputFile << std::setw(10) << std::left << host_a[i] << std::setw(10) << std::left << host_b[i] << std::setw(10) << std::left << host_c[i] << "\n";
		}
		outputFile.close();
		std::cout << "File written successfully.\n";
	} 
	else 
	{
		std::cerr << "Error opening file.\n";
	}
}

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line)
{
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
}

__global__ void kernel_func(float *arr1, float *arr2, float *outp, int length) 
{
	int dimx = length;
	int dimy = length;
	int dimz = length;
	
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < dimx && y < dimy && z < dimz) 
    {
        int index = z * dimx * dimy + y * dimx + x;
        outp[index] = arr1[index] + arr2[index];
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
	
	
	length = 1000000;
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
        host_a[i] = rand_int(min_, max_);
        host_b[i] = rand_int(min_, max_);
        host_c[i] = 0;
    }

	threads_per_block = dim3(16, 8, 4);
	blocks_per_grid = dim3((length + threads_per_block.x - 1) / threads_per_block.x,
						(length + threads_per_block.y - 1) / threads_per_block.y,
						(length + threads_per_block.z - 1) / threads_per_block.z);								
	
	CHECK_CUDA_ERROR(cudaMemcpy(device_a, host_a, sizeof(t) * length, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(device_b, host_b, sizeof(t) * length, cudaMemcpyHostToDevice));
	
	kernel_func<<<blocks_per_grid, threads_per_block>>>(device_a, device_b, device_c, length);
	
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