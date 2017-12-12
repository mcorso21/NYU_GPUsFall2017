/* 
 *  
 *  New York University
 *  GPUs Fall 2017
 *  Steven Adam & Michael Corso
 *  Quantifying the Relationship Between Occupancy and Performance
 * 
 *
 *  Code Explanation:
 *
 *      (1) The first function call is to initDeviceVars() which checks the system for CUDA devices and
 *      chooses the device with the highest Compute Capability (CC). Device-specific variables are then 
 *      populated based on the CC and variables stored in the cudaDeviceProp struct. This information
 *      allows for grid and block dimensions to be constructed according to the specific device this code
 *      is running on.
 *
 *      (2) The code then scales the kernel's parameters based on the device's specifications using the 
 *		user-specified values provided in the program's arguments:
 *
 *          (a) occupancyMethod: 
 *
 *              (0) Blocks per SM: Determines the maximum number of blocks assignable (IE [number of SMs] *
 *              [max blocks assignable to each SM]) and scales it based on the specified targetOccupancy.
 *              The number of threads per block is maxed.
 *
 *              (1) Threads per Block: Determines the maximum number of threads assignable to a block (IE
 *              1024 is common) and scales this based on the specified targetOccupancy. The number of blocks
 *              is equal to ([number of SMs] * [max blocks assignable to each SM]).
 *
 *      	(b) The work being performed by the threads is user-specified in the program's arguments:
 *      
 *          	(0) doubleInt(): 	No memory accesses, simply multiples its thread id by 2
 *
 *          	(1) memoryBound():  Memory-bound vector addition, 3 memory accesses, 1 floating-point addition
 *								    CGMA = 1/3
 *              
 *          	(2) computeBound(): Compute-bound vector math, 3 memory accesses, 90 floating-point operations
 *									CGMA = 90/3 = 30
 *
 *          (c) targetOccupancy: An integer value of 1 - 100 which specifies the percentage of the maximum 
 *              occupancy for this test.
 *              
 *          (d) problemSize: An integer value which specifies the amount of work to be performed by the kernel
 *              
 *
 */

#include <cuda.h>
#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <stdio.h>
#include <time.h> 
#include <algorithm>
#include <vector>

// DEBUG/TEST
#define TESTING false
#define PRINTTIME false

//
void howToUse();

// OCCUPANCY FUNCTIONS
double test_BlocksPerGrid();
double test_ThreadsPerBlock();

// GPU SPEC FUNCTIONS
void initDeviceVars();
void getGPU();
void getMaxBlocksPerSM();
void getMaxWarpsPerSM();

// GPU FUNCTIONS
__global__
void doubleInt (int, int);
__global__
void memoryBound (float *, float *, float *, int, int);
__global__
void computeBound (float *, float *, float *, int, int);

// TEST VARIABLES
int problemSize, occupancyMethod, functionToUse;
double targetOccupancy;

// DEVICE VARIABLES
char * deviceName;
int maxThreadsPerBlock, 
	maxThreadsPerSM, maxBlocksPerSM, maxWarpsPerSM, numSMs, 
	maxThreadsPerGrid, 
	compCapMajor, compCapMinor;

int main(int argc, char * argv[]) {

	// HOW TO USE
    if(argc != 5) howToUse();

	// UPDATE DEVICE VARIABLES
	initDeviceVars();

    // OUTPUT DEVICE-SPECIFIC VALUES
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
    functionToUse = (int) atoi(argv[2]);
    if (functionToUse > 2)
        howToUse();
    targetOccupancy = ((double) (atoi(argv[3]) / 100.0));
    if (targetOccupancy > 1.0) targetOccupancy = 1.0;
    if (targetOccupancy == 0.0) targetOccupancy = 0.01;
    problemSize = (int) atoi(argv[4]);

    // MAX BLOCKS THAT CAN RUN SIMULTANEOUSLY
    if (occupancyMethod == 0) {
        test_BlocksPerGrid();
    }
    // MAX THREADS PER BLOCK
    else if (occupancyMethod == 1) {
        test_ThreadsPerBlock();
    }
    else {
        printf("\nNot an acceptable occupancyMethod!\n");
        howToUse();
    }

    return 0;
}

// BLOCKS PER SM / TOTAL BLOCKS IN THE KERNEL (USES MAX NUMBER OF THREADS PER BLOCK)
double test_BlocksPerGrid() {

	// NUMBER OF BLOCKS
    int totalBlocks = ((numSMs * maxBlocksPerSM) * targetOccupancy);
    if (totalBlocks < 1) totalBlocks = 1;
    
    // ATTEMPT TO DISTRIBUTE THREADS EVENLY 
    int threadsPerBlock = (maxThreadsPerSM * (numSMs / totalBlocks));
    while (threadsPerBlock % 32 != 0) threadsPerBlock -= 1;
    if (threadsPerBlock > maxThreadsPerBlock) threadsPerBlock = maxThreadsPerBlock;
    if (threadsPerBlock < 1) threadsPerBlock = 32;

    // TOTAL NUMBER OF THREADS IN THE GRID
    int totalThreads = totalBlocks * threadsPerBlock;

    dim3 dimGrid(totalBlocks, 1, 1);                       
    dim3 dimBlock(threadsPerBlock, 1, 1);

    if (TESTING) printf("\ntest_MaxBlocksPerSM running with:\n\ttotalBlocks\t%d\t%d%%\n\tblockSize\t%d\t%.01f%%\n", 
        totalBlocks, ((int) (targetOccupancy * 100)), threadsPerBlock, (((float) threadsPerBlock / (float) maxThreadsPerBlock) * 100));

    // ARRAYS FOR PERFORMING VECTOR MATH
    float * in1 = (float *) calloc((problemSize), sizeof(float));
    float * in2 = (float *) calloc((problemSize), sizeof(float));
    float * out = (float *) calloc((problemSize), sizeof(float));
    float * in1D; float * in2D; float * outD;

    for (int i = 0; i < problemSize; i++) {
        in1[i] = (i * 0.99);
        in2[i] = ((problemSize - i - 1) * 0.99);
        out[i] = -1;
    }

    cudaMalloc((void **) &in1D, problemSize);  
    cudaMemcpy(in1D, in1, problemSize, cudaMemcpyHostToDevice);   
    cudaMalloc((void **) &in2D, problemSize);  
    cudaMemcpy(in2D, in2, problemSize, cudaMemcpyHostToDevice);  
    cudaMalloc((void **) &outD, problemSize);  
    cudaMemcpy(outD, out, problemSize, cudaMemcpyHostToDevice);  

    // INITIALIZE TIMER BEFORE CALLING KERNEL
    // clock_t start = clock();
    if (functionToUse == 0) {
        doubleInt<<<dimGrid, dimBlock>>>(problemSize, totalThreads);
    }
    else if (functionToUse == 1) {
        memoryBound<<<dimGrid, dimBlock>>>(in1D, in2D, outD, problemSize, totalThreads);
    }
    else if (functionToUse == 2) {
        computeBound<<<dimGrid, dimBlock>>>(in1D, in2D, outD, problemSize, totalThreads);
    }

    // SYNC DEVICE AND GET TIME TAKEN
    cudaDeviceSynchronize();
    // clock_t end = clock();
    // double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

    // CLEANUP
    free(in1); free(in2); free(out);
    cudaFree(in1D); cudaFree(in2D); cudaFree(outD);

    return time_taken;
}

// THREADS PER BLOCK (USES MAX NUMBER OF BLOCKS)
double test_ThreadsPerBlock() {

	// NUMBER OF THREADS PER BLOCK
    int threadsPerBlock = (maxThreadsPerBlock * targetOccupancy);
    while (threadsPerBlock % 32 != 0) threadsPerBlock -= 1;
    if (threadsPerBlock < 1) threadsPerBlock = 1;

    // NUMBER OF BLOCKS
    int totalBlocks = (maxThreadsPerSM / threadsPerBlock) * numSMs;
    if (totalBlocks < 1) totalBlocks = 1;

    // TOTAL NUMBER OF THREADS IN THE GRID
    int totalThreads = totalBlocks * threadsPerBlock;

    dim3 dimGrid(totalBlocks, 1, 1);                       
    dim3 dimBlock(threadsPerBlock, 1, 1);

    if (TESTING) printf("\ntest_ThreadsPerBlock running with:\n\ttotalBlocks\t%d\t100%%\n\tblockSize\t%d\t%d%%\n", 
        totalBlocks, threadsPerBlock, ((int) (targetOccupancy * 100)));

    // ARRAYS FOR PERFORMING VECTOR MATH
    float * in1 = (float *) calloc((problemSize), sizeof(float));
    float * in2 = (float *) calloc((problemSize), sizeof(float));
    float * out = (float *) calloc((problemSize), sizeof(float));
    float * in1D; float * in2D; float * outD;

    for (int i = 0; i < problemSize; i++) {
        in1[i] = (i * 0.99);
        in2[i] = ((problemSize - i - 1) * 0.99);
        out[i] = -1;
    }

    cudaMalloc((void **) &in1D, problemSize);  
    cudaMemcpy(in1D, in1, problemSize, cudaMemcpyHostToDevice);   
    cudaMalloc((void **) &in2D, problemSize);  
    cudaMemcpy(in2D, in2, problemSize, cudaMemcpyHostToDevice);  
    cudaMalloc((void **) &outD, problemSize);  
    cudaMemcpy(outD, out, problemSize, cudaMemcpyHostToDevice);  

    // INITIALIZE TIMER BEFORE CALLING KERNEL
    clock_t start = clock();
    if (functionToUse == 0) {
        doubleInt<<<dimGrid, dimBlock>>>(problemSize, totalThreads);
    }
    else if (functionToUse == 1) {
        memoryBound<<<dimGrid, dimBlock>>>(in1D, in2D, outD, problemSize, totalThreads);
    }
    else if (functionToUse == 2) {
        computeBound<<<dimGrid, dimBlock>>>(in1D, in2D, outD, problemSize, totalThreads);
    }

    // SYNC DEVICE AND GET TIME TAKEN
    cudaDeviceSynchronize();
    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

    // CLEANUP
    free(in1); free(in2); free(out);
    cudaFree(in1D); cudaFree(in2D); cudaFree(outD);

    return time_taken;
}

// SIMPLE FP MULTIPLICATION, NO MEMORY ACCESS
__global__
void doubleInt (int N, int totalThreads) {

	int id = (blockIdx.x * blockDim.x) + threadIdx.x;
    int val = id;

	while (id < N) {
        val = id;
        val *= 2;
        id += totalThreads;
	}
}

// MEMORY BOUND VECTOR
__global__
void memoryBound (float * in1, float * in2, float * out, int N, int totalThreads) {

	int id = (blockIdx.x * blockDim.x) + threadIdx.x;

	while (id < N) {
		float t1 = in1[id];
		float t2 = in2[id];
		
		out[id] = t1 + t2;

        id += totalThreads;
	}
}

__global__
void computeBound (float * in1, float * in2, float * out, int N, int totalThreads) {

	int id = (blockIdx.x * blockDim.x) + threadIdx.x;

	while (id < N) {
        float t1 = in1[id];
        float t2 = in2[id];
        float tt = t1 + t2;
		for (int i = 0; i < 10; i++)
			tt *= 3.14 * 2.718 / .57721 - 4.6692 + 1.61803 * 131.7 - 530.1874 / 51.9;
		
		out[id] = tt;
        id += totalThreads;
	}
}

// MAIN FUNCTION CALL TO GET DEVICE-SPECIFIC DATA
void initDeviceVars() {

	getGPU();
    getMaxBlocksPerSM();
    getMaxWarpsPerSM();
}

// SET DEVICE WITH HIGHEST COMPUTE CAPABILITY
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

    // WHICH DEVICE HAS HIGHEST COMPUTE CAPABILITY
    for (int i = 0; i < dev_count; i++) {
        cudaGetDeviceProperties(&dev_prop, i);
        if ((dev_prop.major > maxCCmajor) || ((dev_prop.major == maxCCmajor) && (dev_prop.minor > maxCCminor))) {
            deviceToUse = i;
            maxCCmajor = dev_prop.major;
    		maxCCminor = dev_prop.minor;
        }
    }

    // SET DEVICE/DEVICE-SPECIFIC VARIABLES
    cudaGetDeviceProperties(&dev_prop, deviceToUse);
    cudaSetDevice(deviceToUse);
    deviceName = &dev_prop.name[0];
    compCapMajor = maxCCmajor;
    compCapMinor = maxCCminor;
    maxThreadsPerGrid = dev_prop.maxGridSize[0];
    numSMs = dev_prop.multiProcessorCount;
    maxThreadsPerBlock = dev_prop.maxThreadsPerBlock;
}

// GET MAX NUMBER OF BLOCKS ASSIGNABLE TO AN SM
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

// GET MAX NUMBER OF WARPS AND THREADS THAT CAN RUN ON AN SM
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

    fprintf( stderr, "\nUsage: './occupancy [occupancyMethod] [functionToUse] [targetOccupancy] [problemSize]'");
    fprintf( stderr, "\n\tOccupancy Method:\n\t\t0: %% of max blocks that can run simultaneously\n\t\t1: %% of max threads per block");
    fprintf( stderr, "\n\tFunction to Use:\n\t\t0: doubleInt\n\t\t1: memoryBound\n\t\t2: computeBound");
    fprintf( stderr, "\n\n\tIE: './occupancy 0 0 75 100000' runs the kernel with doubleInt() and 75%% of max blocks simultaneously assignable to all SMs and a problem size of 100,000");

    exit( 1 );
}