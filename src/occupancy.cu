/* 
 *  
 *  New York University
 *  GPUs Fall 2017
 *  Steven Adam & Michael Corso
 *  Quantifying the Relationship Between Occupancy and Performance
 *
 * 
 */

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

#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <iostream>
#include <iomanip>
#include <locale.h>

using std::cout;
using std::setw;

// DEBUG/TESTING
bool testing = true;

// CPU FUNCTIONS
void initDeviceVars();
void getGPU();
void test_MaxBlocksPerSM();
void getMaxBlocksPerSM();
void getMaxWarpsPerSM();

// GPU FUNCTIONS
__global__
void doubleInt (int N, int blockSize);

// TEST VARIABLES
int problemSize;

// DEVICE VARIABLES
char * deviceName;
int maxThreadsPerBlock, 
	maxThreadsPerSM, maxBlocksPerSM, maxWarpsPerSM, numSMs, 
	maxThreadsPerGrid, 
	compCapMajor, compCapMinor;

// DEVICE CONSTANTS


int main(int argc, char * argv[]) {

	setlocale(LC_NUMERIC, "");

	// HOW TO USE
    if(argc != 2) {
        fprintf( stderr, "Usage: './occupancy [percentTargetOccupancy]'\n" );
        fprintf( stderr, "\tIE: './occupancy 75' runs the kernel with 75%% occupancy.\n" );
        exit( 1 );
    }

	// UPDATE DEVICE VARIABLES
	initDeviceVars();

    if (testing) {
        printf("\nGPU Info:\n\t%-15s %s\n\t%-15s %d.%d\n\t%-15s %'d\n\t%-15s %'d\n\t%-15s %'d\n\t%-15s %'d\n", 
            "Device ID", deviceName,
            "Compute C.", compCapMajor, compCapMinor, 
            "Grid Size", maxThreadsPerGrid,
            "Block Size", maxThreadsPerBlock,
            "# SMs", numSMs,
            "# Warps", maxWarpsPerSM
            );
    }

    problemSize = 10;

    int targetOccupancy = (int) atoi(argv[1]); // percent ex: 100% 90%

	// Tests saturate:
	// 		(1) warps (per SM or grid or both), 
	// 		(2) threads per block, 
	// 		(3) threads per SM, and 
	// 		(4) blocks per SM
	
	test_MaxBlocksPerSM();
	
    return 0;
}

// TEST OCCUPANCY BY MAXING OUT BLOCKS FOR EVERY SM
void test_MaxBlocksPerSM() {

	int totalBlocks = (numSMs * maxBlocksPerSM);

    dim3 dimGrid(totalBlocks, 1, 1);                       
    // dim3 dimBlock(ceil((problemSize / totalBlocks)), 1, 1); // THIS WAS CRASHING

	// doubleInt<<<dimGrid, dimBlock>>>(problemSize, (problemSize / totalBlocks));
	// cudaDeviceSynchronize();
}

__global__
void doubleInt (int N, int blockSize) {

	int id = (blockIdx.x * blockDim.x) + threadIdx.x;

	id * 2.0;

	while ((id + blockSize) < N) {
		id += blockSize;
		id * 2.0;
	}
}

void initDeviceVars() {

	getGPU();
    getMaxBlocksPerSM();
    getMaxWarpsPerSM();
}

void getGPU() {

    int dev_count, deviceToUse, maxCCmajor, maxCCminor;
    dev_count = deviceToUse = maxCCmajor = maxCCminor = 0;
    
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
        if ((dev_prop.major > maxCCmajor) || ((dev_prop.major == maxCCmajor) && (dev_prop.minor > maxCCminor))) {
            deviceToUse = i;
            maxCCmajor = dev_prop.major;
    		maxCCminor = dev_prop.minor;
        }
    }

    cudaGetDeviceProperties(&dev_prop, deviceToUse);

    cudaSetDevice(deviceToUse);
    deviceName = &dev_prop.name[0];
    compCapMajor = maxCCmajor;
    compCapMinor = maxCCminor;
    maxThreadsPerGrid = dev_prop.maxGridSize[0];
    numSMs = dev_prop.multiProcessorCount;
    maxThreadsPerBlock = dev_prop.maxThreadsPerBlock;
}

// MAX NUMBER OF BLOCKS ASSIGNABLE TO AN SM
void getMaxBlocksPerSM() {

	if (compCapMajor == 2) maxBlocksPerSM = 8;
	else if (compCapMajor == 3) maxBlocksPerSM = 16;
	else if ((compCapMajor == 5) || (compCapMajor == 6)) maxBlocksPerSM = 32;
	else {
		printf("\n No max blocks settings for Compute Capability %d.%d\n", 
			compCapMajor, compCapMinor);
		exit(0);
	}
}

// MAX NUMBER OF WARPS AND THREADS THAT CAN RUN ON AN SM
void getMaxWarpsPerSM() {

	if (compCapMajor == 2) maxWarpsPerSM = 48;
	else if ((compCapMajor == 3) || (compCapMajor == 5)) maxWarpsPerSM = 64;
	else if (compCapMajor == 6) {
		if (compCapMinor == 2) maxWarpsPerSM = 128;
		else maxWarpsPerSM = 64;
	}
	else {
		printf("\n No max warp settings for Compute Capability %d.%d\n", 
			compCapMajor, compCapMinor);
		exit(0);
	}
	// ASSIGN MAX THREADS PER SM
	maxThreadsPerSM = (maxWarpsPerSM * 32);
}


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