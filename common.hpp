#ifndef COMMON_HPP
#define COMMON_HPP

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

typedef float T;
typedef int I;

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

void write_output_to_file(I* host_a, I* host_b, I* host_c, std::string fileName, int length) 
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
#endif
