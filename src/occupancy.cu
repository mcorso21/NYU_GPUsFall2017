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

// DEBUG/TEST
#define TESTING true

//
void howToUse();

// OCCUPANCY FUNCTIONS
void test_BlocksPerSM();
void test_ThreadsPerBlock();
void test_ThreadsPerSM();
void test_WarpsPerSM();
void test_threadsPerBlockPerKernel();

// GPU SPEC FUNCTIONS
void initDeviceVars();
void getGPU();
void getMaxBlocksPerSM();
void getMaxWarpsPerSM();

// GPU FUNCTIONS
__global__
void doubleInt (int N, int blockSize);

// TEST VARIABLES
int problemSize, occupancyMethod;
double targetOccupancy;

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
    if(argc != 4) howToUse();

	// UPDATE DEVICE VARIABLES
	initDeviceVars();

    if (TESTING) {
        printf("\nGPU Info:\n\t%-15s %s\n\t%-15s %d.%d\n\t%-15s %d\n\t%-15s %d\n\t%-15s %d\n\t%-15s %d\n", 
            "Device ID", deviceName,
            "Compute C.", compCapMajor, compCapMinor, 
            "Grid Size", maxThreadsPerGrid,
            "Block Size", maxThreadsPerBlock,
            "# SMs", numSMs,
            "# Warps", maxWarpsPerSM
            );
    }

    // GET USER-SPECIFIED VARIABLES
    occupancyMethod = (int) atoi(argv[1]);
    targetOccupancy = ((double) (atoi(argv[2]) / 100.0));
    if (targetOccupancy > 1.0) targetOccupancy = 1.0;
    if (targetOccupancy == 0.0) targetOccupancy = 0.01;
    problemSize = (int) atoi(argv[3]);
    
    // MAX BLOCKS THAT CAN RUN SIMULTANEOUSLY
    if (occupancyMethod == 0) {
        test_BlocksPerSM();
    }
    // MAX THREADS PER BLOCK
    else if (occupancyMethod == 1) {
        test_ThreadsPerBlock();
    }
    // MAX THREADS PER SM
    else if (occupancyMethod == 2) {
        test_ThreadsPerSM();
    }
    // MAX WARPS PER SM
    else if (occupancyMethod == 3) {
        test_WarpsPerSM();
    }
    // THREADS/BLOCK INVERSED WITH BLOCKS/KERNEL
    else if (occupancyMethod == 4) {
        test_threadsPerBlockPerKernel();
    }
    else {
        printf("\nNot an acceptable occupancyMethod!\n");
        howToUse();
    }
		
    return 0;
}

// THREADS/BLOCK INVERSED WITH BLOCKS/KERNEL
// THIS ACTS LIKE A SEESAW: TOTALBLOCKS GOES UP AS THREADS PER BLOCK GOES DOWN AND VICE VERSA
void test_threadsPerBlockPerKernel() {

    int totalBlocks = ((numSMs * maxBlocksPerSM) * targetOccupancy);
    int threadsPerBlock = maxThreadsPerBlock * (1.0 - targetOccupancy);
    if (threadsPerBlock <= 0) threadsPerBlock = 1;
    if (totalBlocks <= 0) totalBlocks = 1;

    dim3 dimGrid(totalBlocks, 1, 1);                       
    dim3 dimBlock(threadsPerBlock, 1, 1);

    if (TESTING) printf("\ntest_threadsPerBlockPerKernel running with:\n\ttotalBlocks = %d\n\tblockSize = %d\n", 
        totalBlocks, threadsPerBlock);

    doubleInt<<<dimGrid, dimBlock>>>(problemSize, threadsPerBlock);
    cudaDeviceSynchronize();
}

// TOTAL BLOCKS
void test_BlocksPerSM() {

    int totalBlocks = ((numSMs * maxBlocksPerSM) * targetOccupancy);
    int threadsPerBlock = maxThreadsPerBlock;

    dim3 dimGrid(totalBlocks, 1, 1);                       
    dim3 dimBlock(threadsPerBlock, 1, 1);

    if (TESTING) printf("\ntest_MaxBlocksPerSM running with:\n\ttotalBlocks = %d\n\tblockSize = %d\n", 
        totalBlocks, threadsPerBlock);

    doubleInt<<<dimGrid, dimBlock>>>(problemSize, threadsPerBlock);
    cudaDeviceSynchronize();
}

// THREADS PER BLOCK (USES MAX NUMBER OF BLOCKS)
void test_ThreadsPerBlock() {

    int totalBlocks = (numSMs * maxBlocksPerSM);
    int threadsPerBlock = (maxThreadsPerBlock * (targetOccupancy));

    dim3 dimGrid(totalBlocks, 1, 1);                       
    dim3 dimBlock(threadsPerBlock, 1, 1);

    if (TESTING) printf("\ntest_ThreadsPerBlock running with:\n\ttotalBlocks = %d\n\tblockSize = %d\n", 
        totalBlocks, threadsPerBlock);

    doubleInt<<<dimGrid, dimBlock>>>(problemSize, threadsPerBlock);
    cudaDeviceSynchronize();
}

// THREADS PER SM

/* MAX NUMBER OF BLOCKS, REDUCE THREADS IN BLOCK TO LESSEN MAX THREADS IN AN SM
 * ISN'T THIS THE SAME RESULT AS test_ThreadsPerBlock? Redundant?
 */

void test_ThreadsPerSM() {

    // int totalBlocks = (numSMs * maxBlocksPerSM);
    
    // dim3 dimGrid(totalBlocks, 1, 1);                       
    // dim3 dimBlock(maxThreadsPerBlock, 1, 1);

    // if (TESTING) printf("\test_ThreadsPerSM running with:\n\ttotalBlocks = %d\n\tblockSize = %d\n", 
    //     totalBlocks, maxThreadsPerBlock);

    // doubleInt<<<dimGrid, dimBlock>>>(problemSize, threadsPerBlock);
    // cudaDeviceSynchronize();
}

// WARPS PER SM
/* SAME QUESTION HERE AS FOR THREADSPERSM ... 
 * MAX NUMBER OF BLOCKS, REDUCE THREADS IN BLOCK TO LESSEN MAX THREADS IN AN SM
 * ISN'T THIS THE SAME RESULT AS test_ThreadsPerBlock? Redundant?
 */

void test_WarpsPerSM() {

    // int totalBlocks = (numSMs * maxBlocksPerSM);
    // int threadsPerBlock = maxThreadsPerBlock;

    // dim3 dimGrid(totalBlocks, 1, 1);                       
    // dim3 dimBlock(threadsPerBlock, 1, 1);

    // if (TESTING) printf("\test_WarpsPerSM running with:\n\ttotalBlocks = %d\n\tblockSize = %d\n", 
    //     totalBlocks, threadsPerBlock);

    // doubleInt<<<dimGrid, dimBlock>>>(problemSize, threadsPerBlock);
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

void howToUse() {

    fprintf( stderr, "\nUsage: './occupancy [occupancyMethod] [targetOccupancy] [problemSize]'");
    fprintf( stderr, "\n\tOccupancy Method:\n\t0: %% of max blocks that can run simultaneously\n\t1: %% of max threads per block\n\t2: %% of max threads per SM\n\t3: %% of max warps per SM\n\t4: %% will inversely scale number of blocks with threads per block");
    fprintf( stderr, "\n\n\tIE: './occupancy 3 75 100000' runs the kernel with 75%% of max warps per SM with a problem size of 100,000");

    exit( 1 );
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