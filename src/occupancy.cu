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

bool testing = true;

void getGPU(int *);

int main() {

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

    // dim3 dimGrid(ceil (N / blockSize), 1, 1);                       
    // dim3 dimBlock((int) blockSize, 1, 1);

    return 0;
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
