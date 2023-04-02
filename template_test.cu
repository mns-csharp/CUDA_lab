#include "template.hpp"

const int min_ = 0;
const int max_ = 10;

__host__ void init_data_(CudaManager<float>& manager) 
{
    for (int i = 0; i < manager.get_length(); ++i) 
	{
        manager.set_host_a(i, rand_int(min_, max_));
        manager.set_host_b(i, rand_int(min_, max_));
        manager.set_host_c(i, 0);
    }
}

__global__ void vector_add(float *arr1, float *arr2, float *outp, int length) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < length) 
    {
        outp[tid] = arr1[tid] + arr2[tid];
    }
}

int main()
{
    init_rand();
	
    CudaManager<float> manager;
    manager.allocate_mem(1000000);
    manager.init_data(init_data_);
	manager.set_thread_dim(16, 16, 1);
    //manager.display_host_data();
	manager.launch_kernel(vector_add);
	manager.display_elapsed_time();
    manager.write_output_to_file("template_output.txt");
    manager.free_mem();
}


