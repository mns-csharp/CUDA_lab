#ifndef TEMPLATE_HPP
#define TEMPLATE_HPP

#include <iostream>
#include <cstdio>

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
    template<class t> friend void init_data_func(CudaManager& manager);
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
        kernel_func<<<blocks_per_grid, 
threads_per_block>>>(device_a, device_b, device_c, length);
        cudaDeviceSynchronize();
        cudaMemcpy(device_a, host_a, length, cudaMemcpyDeviceToHost);
        cudaMemcpy(device_b, host_b, length, cudaMemcpyDeviceToHost);
        cudaMemcpy(device_c, host_c, length, cudaMemcpyDeviceToHost);
    }
    time_t get_elapsed_time(){}
    void display_data()
    {
        std::cout<<"host data:\n";
        for(int i=0 ; i<length ; i++)
        {
            std::cout<<host_a[i]<<", "<<host_b[i]<<", "<<host_c[i]<<"\n";
        }

        std::cout<<"device data:\n";
        for(int i=0 ; i<length ; i++)
        {
            std::cout<<device_a[i]<<", "<<device_b[i]<<", "<<device_c[i]<<"\n";
        }
    }
};
#endif
