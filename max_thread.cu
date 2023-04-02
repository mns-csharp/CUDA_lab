#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        printf("Device %d: %s\n", i, deviceProp.name);
        printf("  Maximum threads per block: %d\n", deviceProp.maxThreadsPerBlock);
    }
	
	
	threads_per_block = dim3(16, 16, 16);
	blocks_per_grid = dim3((length + threads_per_block.x - 1) / threads_per_block.x,
						(length + threads_per_block.y - 1) / threads_per_block.y,
						(length + threads_per_block.z - 1) / threads_per_block.z);	
	
	printf("blocks_per_grid dimensions: (%d, %d, %d)\n", blocks_per_grid.x, blocks_per_grid.y, blocks_per_grid.z);
	
    return 0;
}
