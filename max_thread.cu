#include <stdio.h>
#include <iostream>
#include <cuda_runtime_api.h>

int main() 
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for (int i = 0; i < deviceCount; i++) 
	{
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        printf("Device %d: %s\n", i, deviceProp.name);
        printf("  Maximum threads per block: %d\n", deviceProp.maxThreadsPerBlock);
    }
	
    dim3 maxGridSize;
    int deviceID = 0; // use device 0
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceID);
    maxGridSize.x = deviceProp.maxGridSize[0];
    maxGridSize.y = deviceProp.maxGridSize[1];
    maxGridSize.z = deviceProp.maxGridSize[2];
    std::cout << "Maximum blocks per grid: " << maxGridSize.x * maxGridSize.y * maxGridSize.z << std::endl;
	
	////////////////////////////////////////////
	int length = 1000000;
	dim3 threads_per_block(16, 16, 16);
	dim3 blocks_per_grid((length + threads_per_block.x - 1) / threads_per_block.x,
						 (length + threads_per_block.y - 1) / threads_per_block.y,
						 (length + threads_per_block.z - 1) / threads_per_block.z);	
	printf("blocks_per_grid dimensions: (%d, %d, %d)\n", blocks_per_grid.x, blocks_per_grid.y, blocks_per_grid.z);
    
	return 0;
}
