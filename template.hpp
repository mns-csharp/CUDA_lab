#ifndef TEMPLATE_HPP
#define TEMPLATE_HPP

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

//#CHECK(call)


template<typename t>
class CudaManager
{
private:
    t * device_a;
    t * device_b;
    t * device_c;
    t * host_a;
    t * host_b;
    t * host_c;
    int length;
    dim3 threads_per_block;
    dim3 blocks_per_grid;
	std::chrono::high_resolution_clock::time_point start_;
	std::chrono::high_resolution_clock::time_point stop_;

public:
    int get_length() const
	{
		return length;
	}
	int get_host_a(int i)
	{
		return host_a[i];
	}
	int get_host_b(int i)
	{
		return host_b[i];
	}
	int get_host_c(int i)
	{
		return host_c[i];
	}
	int get_device_a(int i)
	{
		return device_a[i];
	}
	int get_device_b(int i)
	{
		return device_b[i];
	}
	int get_device_c(int i)
	{
		return device_c[i];
	}
	
	void set_host_a(int i, t value)
	{
		host_a[i] = value;
	}
	void set_host_b(int i, t value)
	{
		host_b[i] = value;
	}
	void set_host_c(int i, t value)
	{
		host_c[i] = value;
	}
	void set_device_a(int i, t value)
	{
		device_a[i] = value;
	}
	void set_device_b(int i, t value)
	{
		device_b[i] = value;
	}
	void set_device_c(int i, t value)
	{
		device_c[i] = value;
	}
	
    void allocate_mem(int n)
    {
        length = n;
        host_a = (t *) malloc(sizeof(t) * length);
        host_b = (t *) malloc(sizeof(t) * length);
        host_c = (t *) malloc(sizeof(t) * length);
        cudaMalloc((void**)&device_a, sizeof(t) * length);
		cudaMalloc((void**)&device_b, sizeof(t) * length);
		cudaMalloc((void**)&device_c, sizeof(t) * length);
    }
    void init_data(void (*init_data_func)(CudaManager&))
    {
        init_data_func(*this);
    }
    void set_thread_dim(int x, int y, int z)
    {
         threads_per_block = dim3(x, y, z);
         blocks_per_grid = dim3((length + threads_per_block.x - 1) / threads_per_block.x,
                                (length + threads_per_block.y - 1) / threads_per_block.y,
                                (length + threads_per_block.z - 1) / threads_per_block.z);
    }
    void free_mem()
    {
        cudaFree(device_a);
        cudaFree(device_b);
        cudaFree(device_c);
        free(host_a);
        free(host_b);
        free(host_c);
    }
    void verify_result()
    {
    }
    void launch_kernel(void (*kernel_func)(t*, t*, t*, int))
    {
        cudaMemcpy(device_a, host_a, sizeof(t) * length, cudaMemcpyHostToDevice);
		cudaMemcpy(device_b, host_b, sizeof(t) * length, cudaMemcpyHostToDevice);
		
		// start measuring time
		start_ = std::chrono::high_resolution_clock::now();		
		
		kernel_func<<<blocks_per_grid, threads_per_block>>>(device_a, device_b, device_c, length);
		cudaDeviceSynchronize();
		
		// stop measuring time
        stop_ = std::chrono::high_resolution_clock::now();
		
		cudaMemcpy(host_c, device_c, sizeof(t) * length, cudaMemcpyDeviceToHost);
    }
    std::chrono::microseconds::rep get_elapsed_time()
	{
		std::chrono::microseconds duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_ - start_);
        return duration.count();
	}
	void display_elapsed_time()
	{
		std::cout << "Kernel execution time: " << this->get_elapsed_time() << " microseconds.\n";
	}
    void display_host_data()
    {
		if(host_a!=nullptr && host_b!=nullptr && host_c!=nullptr)
        {
			std::cout<<"host data:\n";
			for(int i=0 ; i<length ; i++)
			{
				std::cout<<host_a[i]<<", "<<host_b[i]<<", "<<host_c[i]<<"\n";
			}
		}
		else
		{
			std::cout<<"host data not found";
			if(host_a==nullptr) std::cout<<"host_a is empty"; 
			if(host_b==nullptr) std::cout<<"host_b is empty"; 
			if(host_c==nullptr) std::cout<<"host_c is empty";
		}
	}
	void write_output_to_file(std::string fileName) 
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
};
#endif
