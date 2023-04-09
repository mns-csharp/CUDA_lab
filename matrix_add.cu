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

void print_dim3(std::string text, dim3 data)
{
    std::cout << "\n";
    std::cout << text << " ";
    std::cout << data.x <<" * "<< data.y<<" * "<< data.z << " = ";
	std::cout << data.x * data.y * data.z;
	std::cout << "\n";
}

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file, const int line)
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

__global__ void kernel_func(float *A, float *B, float *C, int N) 
{
	int dimx = N;
	int dimy = N;
	int dimz = N;
	
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < dimx && j < dimy && k < dimz) 
    {
        int index = k * dimx * dimy + j * dimx + i;
        C[index] = A[index] + B[index];
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
	
	
	length = 3;
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
	
    kernel_func<<<blocks_per_grid, threads_per_block>>>(device_a, device_b, device_c, 100);
	
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
