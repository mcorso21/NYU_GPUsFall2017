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
bool testing = false;

// CPU FUNCTIONS
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
int numThreadsPerSM, numBlocksPerSM, numWarpsPerSM, numSMs, compCapMajor, compCapMinor;

// DEVICE CONSTANTS
int numThreadsPerBlock = 1024;

int main(int argc, char * argv[]) {

	setlocale(LC_NUMERIC, "");

	    // HOW TO USE
    if(argc != 2) {
        fprintf( stderr, "Usage: './occupancy [percentTargetOccupancy]'\n" );
        fprintf( stderr, "\tIE: './occupancy 75' runs the kernel with 75%% occupancy.\n" );
        exit( 1 );
    }

    // DEVICE, MAX BLOCK SIZE, #SMS, MAX GRID SIZE, MAX THREADS, WARP SIZE, # REGISTERS PER BLOCK, MAJOR, MINOR
    // int devInfo [9];
    getGPU();

    // IF MULTIPLE DEVICES PRESENT, USE DEVICE WITH LARGEST BLOCK SIZE
    // cudaSetDevice(devInfo[0]);

    // if (testing) {
    //     printf("\nGPU Info:\n\t%-15s %'d\n\t%-15s %d.%d\n\t%-15s %'d\n\t%-15s %'d\n\t%-15s %'d\n\t%-15s %'d\n\t%-15s %'d\n\t%-15s %'d\n", 
    //         "Device ID", devInfo[0],
    //         "Compute C.", devInfo[7], devInfo[8], 
    //         "Grid Size", devInfo[3],
    //         "Block Size", devInfo[1],
    //         "Max Threads", devInfo[4],
    //         "# SMs", devInfo[2],
    //         "Warp Size", devInfo[5],
    //         "# Registers", devInfo[6]
    //         );
    // }

 //    /****************
 //    * Device specific
 //    *****************/
 //    int maxWarps = 64;
 //    int threadsPerWarp = devInfo[5]; // 32
 //    int maxThreadsPerBlock = devInfo[1];

 //    /***************************
 //    * Dynamic thread computation
 //    ****************************/
 //    /*
 //    * Given 90% target occupancy
 //    * max 2048 threads * 90% = 1844
 //    * Split into 2 blocks on the grid 
 //    * Each block handles 922 threads
 //    */
    // int targetOccupancy = (int) atoi(argv[1]); // percent ex: 100% 90%
 //    // // ***** occupancy above 50% is inaccurate, need to figure this out
 //    int numThreads = ceil((targetOccupancy/100.0) * maxWarps * threadsPerWarp); // threads to achieved target occupancy
 //    int numBlocksPerGrid = ceil((float) numThreads/maxThreadsPerBlock);
 //    int blockSize = ceil((float) numThreads/numBlocksPerGrid);

 //    dim3 dimGrid(1, 1, 1);                       
 //    dim3 dimBlock(blockSize, numBlocksPerGrid, 1);

 //    problemSize = (maxWarps * threadsPerWarp) * 1000; // 2048

 // //    doubleInt<<<dimGrid, dimBlock>>>(problemSize, blockSize);
	// // cudaDeviceSynchronize();

    problemSize = 10;

    int targetOccupancy = (int) atoi(argv[1]); // percent ex: 100% 90%

	// UPDATE DEVICE VARIABLES
    getMaxBlocksPerSM();
    getMaxWarpsPerSM();


	// Tests saturate:
	// 		(1) warps (per SM or grid or both), 
	// 		(2) threads per block, 
	// 		(3) threads per SM, and 
	// 		(4) blocks per SM
	
	test_MaxBlocksPerSM();
	
    return 0;
}

// MAX NUMBER OF BLOCKS ASSIGNABLE TO AN SM
void getMaxBlocksPerSM() {

	if (compCapMajor == 2) numBlocksPerSM = 8;
	else if (compCapMajor == 3) numBlocksPerSM = 16;
	else if ((compCapMajor == 5) || (compCapMajor == 6)) numBlocksPerSM = 32;
	else {
		printf("\n No max blocks settings for Compute Capability %d.%d\n", 
			compCapMajor, compCapMinor);
		exit(0);
	}
}

// MAX NUMBER OF WARPS AND THREADS THAT CAN RUN ON AN SM
void getMaxWarpsPerSM() {

	if (compCapMajor == 2) numWarpsPerSM = 48;
	else if ((compCapMajor == 3) || (compCapMajor == 5)) numWarpsPerSM = 64;
	else if (compCapMajor == 6) {
		if (compCapMinor == 2) numWarpsPerSM = 128;
		else numWarpsPerSM = 64;
	}
	else {
		printf("\n No max warp settings for Compute Capability %d.%d\n", 
			compCapMajor, compCapMinor);
		exit(0);
	}
	// ASSIGN MAX THREADS PER SM
	numThreadsPerSM = (numWarpsPerSM * 32);
}

// TEST OCCUPANCY BY MAXING OUT BLOCKS FOR EVERY SM
void test_MaxBlocksPerSM() {

	if (testing) 
		printf("Number of SMs: %d\nBlocks per SM: %d", 
			numSMs, numBlocksPerSM);

	int totalBlocks = (numSMs * numBlocksPerSM);

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


    // #if __CUDA_ARCH__ >= 200
    //     printf("Hello thread %d \n", id);
    // #endif
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

    cudaSetDevice(deviceToUse);
    compCapMajor = maxCCmajor;
    compCapMinor = maxCCminor;
}
