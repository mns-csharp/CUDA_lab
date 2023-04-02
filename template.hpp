#ifndef TEMPLATE_HPP
#define TEMPLATE_HPP

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#include <cuda.h>

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
         cudaMalloc(&device_a, length);
         cudaMalloc(&device_b, length);
         cudaMalloc(&device_c, length);
    }
    //friend void init_data_func(CudaManager& manager);
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
        cudaMemcpy(device_a, host_a, length, cudaMemcpyHostToDevice);
        cudaMemcpy(device_b, host_b, length, cudaMemcpyHostToDevice);
        cudaMemcpy(device_c, host_c, length, cudaMemcpyHostToDevice);
        kernel_func<<<blocks_per_grid, threads_per_block>>>(device_a, device_b, device_c, length);
        cudaDeviceSynchronize();
        cudaMemcpy(device_a, host_a, length, cudaMemcpyDeviceToHost);
        cudaMemcpy(device_b, host_b, length, cudaMemcpyDeviceToHost);
        cudaMemcpy(device_c, host_c, length, cudaMemcpyDeviceToHost);
    }
    time_t get_elapsed_time(){}
    void display_host()
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
    /*void display_device()
    {	
		if(device_a!=nullptr)
        {
			std::cout<<"device_a: ";
			for(int i=0 ; i<length ; i++)
			{
				std::cout<<device_a[i]<<", ";
			}
		}
		if(device_b!=nullptr)
        {
			std::cout<<"device_b: ";
			for(int i=0 ; i<length ; i++)
			{
				std::cout<<device_c[i]<<", ";
			}
		}
		if(device_c!=nullptr)
        {
			std::cout<<"device_c: ";
			for(int i=0 ; i<length ; i++)
			{
				std::cout<<device_c[i]<<", ";
			}
		}		
		
		if(device_a==nullptr) std::cout<<"device_a is empty"; 
		if(device_b==nullptr) std::cout<<"device_b is empty"; 
		if(device_c==nullptr) std::cout<<"device_c is empty";
    }*/
};
#endif
