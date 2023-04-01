#include "include/template.hpp"

const int min = 0;
const int min = 10;

__host__ void init_data_(CudaManager<float>& manager) 
{
    for (int i = 0; i < manager.get_length(); ++i) 
	{
        manager.set_host_a(i, rand_int(min, max));
        manager.set_host_b(i, rand_int(min, max));
        manager.set_host_c(i, rand_int(min, max));
    }
}

__global__ void vector_add(float *arr1, float *arr2, float *outp, int n) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) 
    {
        outp[tid] = arr1[tid] + arr2[tid];
    }
}

int main()
{
    init_rand();
	
    CudaManager<float> manager;
    manager.allocate_mem(10);
    manager.init_data(init_data_);
    manager.display_host();
	manager.launch_kernel(vector_add);
    manager.display_host();
    manager.free_mem();
}


