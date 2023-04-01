#include "template.hpp"

__host__ void init_data_func(CudaManager<float>& manager) 
{
    for (int i = 0; i < manager.length; ++i) {
        manager.host_a[i] = rand();
        manager.host_b[i] = rand();
        manager.host_c[i] = rand();
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
    CudaManager<float> manager;
    manager.allocate_mem(10);
    manager.init_data(init_data_func);
    manager.launch_kernel(vector_add);
    manager.display_data();
    manager.free_mem();
}
