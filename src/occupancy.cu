/* 
 *  
 *  New York University
 *  GPUs Fall 2017
 *  Steven Adam & Michael Corso
 *  Quantifying the Relationship Between Occupancy and Performance
 *
 * 
 */

#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <iostream>
#include <iomanip>
#include <locale.h>

using std::cout;
using std::setw;

__global__
void doubleInt (int N, int blockSize);

bool testing = true;

void getGPU(int *);

int main(int argc, char * argv[]) {

	// GOAL: CURRENT WARPS / MAX WARPS PER SM
	/*
		IDEA #1

		Create a block of 2048, which is 100% warp occupancy (2048/32 [warp size] = 64 [max # warps])

		In bash script
			blockSize = 2048
			while (blockSize >= 32)
				Call occupancy.cu with block/grid size
			
		In occupancy.cu
			gridSize = arg
			blockSize = arg

			call kernel function with grid/block size 

	*/

	setlocale(LC_NUMERIC, "");

	    // HOW TO USE
    if(argc != 2) {
        fprintf( stderr, "Usage: './occupancy [percentTargetOccupancy]'\n" );
        exit( 1 );
    }

    // DEVICE, MAX BLOCK SIZE, #SMS, MAX GRID SIZE, MAX THREADS, WARP SIZE, # REGISTERS PER BLOCK, MAJOR, MINOR
    int devInfo [9];
    getGPU(devInfo);

    // IF MULTIPLE DEVICES PRESENT, USE DEVICE WITH LARGEST BLOCK SIZE
    cudaSetDevice(devInfo[0]);

    if (testing) {
        printf("\nGPU Info:\n\t%-15s %'d\n\t%-15s %d.%d\n\t%-15s %'d\n\t%-15s %'d\n\t%-15s %'d\n\t%-15s %'d\n\t%-15s %'d\n\t%-15s %'d\n", 
            "Device ID", devInfo[0],
            "Compute C.", devInfo[7], devInfo[8], 
            "Grid Size", devInfo[3],
            "Block Size", devInfo[1],
            "Max Threads", devInfo[4],
            "# SMs", devInfo[2],
            "Warp Size", devInfo[5],
            "# Registers", devInfo[6]
            );
    }

    /****************
    * Device specific
    *****************/
    int maxWarps = 64;
    int threadsPerWarp = devInfo[5]; // 32
    int maxThreadsPerBlock = devInfo[1];

    /***************************
    * Dynamic thread computation
    ****************************/
    /*
    * Given 90% target occupancy
    * max 2048 threads * 90% = 1844
    * Split into 2 blocks on the grid 
    * Each block handles 922 threads
    */
    int targetOccupancy = (int) atoi(argv[1]); // percent ex: 100% 90%
    int numThreads = ceil((targetOccupancy/100.0) * 64 * 32); // threads to achieved target occupancy
    int numBlocksPerGrid = ceil((float) numThreads/maxThreadsPerBlock);
    int blockSize = ceil((float) numThreads/numBlocksPerGrid);

    dim3 dimGrid(numBlocksPerGrid, 1, 1);                       
    dim3 dimBlock(blockSize, 1, 1);

    int problemSize = maxWarps * threadsPerWarp; // 2048

    doubleInt<<<dimGrid, dimBlock>>>(problemSize, blockSize);

    cudaDeviceSynchronize();
    return 0;
}

__global__
void doubleInt (int N, int blockSize) {

	int id = (blockIdx.x * blockDim.x) + threadIdx.x;

	id * 2.0;

	while ((id + blockSize) < N) {
		id += blockSize;
		id * 2.0;
	}

    printf("Hello thread %d \n", id);
}


// SETS [ GPUID, BLOCKSIZE, #SMs ] TO MAXIMIZE BLOCK SIZE
/* Device property struct:

    struct cudaDeviceProp {
        char name[256];
        size_t totalGlobalMem;
        size_t sharedMemPerBlock;
        int regsPerBlock;
        int warpSize;
        size_t memPitch;
        int maxThreadsPerBlock;
        int maxThreadsDim[3];
        int maxGridSize[3];
        size_t totalConstMem;
        int major;
        int minor;
        int clockRate;
        size_t textureAlignment;
        int deviceOverlap;
        int multiProcessorCount;
        int kernelExecTimeoutEnabled;
        int integrated;
        int canMapHostMemory;
        int computeMode;
        int concurrentKernels;
        int ECCEnabled;
        int pciBusID;
        int pciDeviceID;
        int tccDriver;
    }
*/
void getGPU(int * devInfo) {

    int dev_count, deviceToUse, blockSize, numSMs, 
    	gridSize, maxThreads, warpSize, regsPerBlock;
    dev_count = deviceToUse = blockSize = numSMs = 0;
    warpSize = regsPerBlock = maxThreads = gridSize = 0;

    // GET NUMBER OF DEVICES
    cudaDeviceProp dev_prop;
    cudaGetDeviceCount(&dev_count);
    if (dev_count < 1) {
        printf("No CUDA enabled devices on this system!\n");
        exit(1);
    }

    // WHICH DEVICE HAS LARGEST BLOCK SIZE
    for (int i = 0; i < dev_count; i++) {
        cudaGetDeviceProperties(&dev_prop, i);
        if (dev_prop.maxThreadsPerBlock > blockSize) {
        	gridSize = dev_prop.maxGridSize[0];
            blockSize = dev_prop.maxThreadsPerBlock;
            warpSize = dev_prop.warpSize;
            regsPerBlock = dev_prop.regsPerBlock;
            maxThreads = dev_prop.maxThreadsDim[0];
            numSMs = dev_prop.multiProcessorCount;
            deviceToUse = i;
        }
    }

    devInfo[0] = deviceToUse; 	// GPU ID
    devInfo[1] = blockSize;		// Max block size
    devInfo[2] = numSMs;		// # SMs
    devInfo[3] = gridSize;		// Max grid size
    devInfo[4] = maxThreads; 	// Max # threads
    devInfo[5] = warpSize; 		// Warp size
    devInfo[6] = regsPerBlock;	// # Registers per block
    devInfo[7] = dev_prop.major;
    devInfo[8] = dev_prop.minor;
}
