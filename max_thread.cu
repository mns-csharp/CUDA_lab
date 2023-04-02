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
	
	//for CUDA 10.0 or lower.
    int maxGridSize[3];
	int deviceID = 0; // use device 0
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, deviceID);
	maxGridSize[0] = deviceProp.maxGridSize[0];
	maxGridSize[1] = deviceProp.maxGridSize[1];
	maxGridSize[2] = deviceProp.maxGridSize[2];

	std::cout << "Maximum blocks per grid : total: " << (unsigned long long)maxGridSize[0] * (unsigned long long)maxGridSize[1] * (unsigned long long)maxGridSize[2] << std::endl; 
	std::cout << "  Maximum blocks per grid : x dimension: " << maxGridSize[0] << std::endl;
	std::cout << "  Maximum blocks per grid : y dimension: " << maxGridSize[1] << std::endl;
	std::cout << "  Maximum blocks per grid : z dimension: " << maxGridSize[2] << std::endl;

	/* //for CUDA 10.1 or higher
	int maxGridSize[3];
	int deviceID = 0; // use device 0
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, deviceID);
	cudaDeviceGetAttribute(&maxGridSize[0], cudaDevAttrMaxGridSizeX, deviceID);
	cudaDeviceGetAttribute(&maxGridSize[1], cudaDevAttrMaxGridSizeY, deviceID);
	cudaDeviceGetAttribute(&maxGridSize[2], cudaDevAttrMaxGridSizeZ, deviceID);

	std::cout << "Maximum blocks per grid : total: " << (unsigned long long)maxGridSize[0] * (unsigned long long)maxGridSize[1] * (unsigned long long)maxGridSize[2] << std::endl; 
	std::cout << "  Maximum blocks per grid : x dimension: " << maxGridSize[0] << std::endl;
	std::cout << "  Maximum blocks per grid : y dimension: " << maxGridSize[1] << std::endl;
	std::cout << "  Maximum blocks per grid : z dimension: " << maxGridSize[2] << std::endl;
	*/
	
	////////////////////////////////////////////
	int length = 1000000;
	dim3 threads_per_block(16, 16, 16);
	dim3 blocks_per_grid((length + threads_per_block.x - 1) / threads_per_block.x,
						 (length + threads_per_block.y - 1) / threads_per_block.y,
						 (length + threads_per_block.z - 1) / threads_per_block.z);	
	printf("\nthreads per block : %d\n", 16 * 16 *16);
	printf("\nblocks per grid dimensions : (%d, %d, %d)\n", blocks_per_grid.x, blocks_per_grid.y, blocks_per_grid.z);
    ////////////////////////////////////////////
	
	return 0;
}
