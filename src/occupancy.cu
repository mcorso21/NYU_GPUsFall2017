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

using std::cout;
using std::setw;

bool testing = true;

void getGPU(int *);

int main() {

    // DEVICE, MAX BLOCK SIZE, #SMS
    int devInfo [3];
    getGPU(devInfo);

    // IF MULTIPLE DEVICES PRESENT, USE DEVICE WITH LARGEST BLOCK SIZE
    cudaSetDevice(devInfo[0]);

    if (testing) {
        printf("\nGPU Info:\n\t%-15s %d\n\t%-15s %d\n\t%-15s %d", 
            "Device ID", devInfo[0],
            "Block Size", devInfo[1],
            "# SMs", devInfo[2]
            );

    }

    // dim3 dimGrid(ceil (N / blockSize), 1, 1);                       
    // dim3 dimBlock((int) blockSize, 1, 1);

    return 0;
}


// SETS [ GPUID, BLOCKSIZE, #SMs ] TO MAXIMIZE BLOCK SIZE
void getGPU(int * devInfo) {

    int dev_count, deviceToUse, blockSize, numSMs;
    dev_count = deviceToUse = blockSize = numSMs = 0;

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
            blockSize = dev_prop.maxThreadsPerBlock;
            numSMs = dev_prop.multiProcessorCount;
            deviceToUse = i;
        }
    }

    // SET GPUID TO USE, ITS BLOCK SIZE, AND ITS SM COUNT
    devInfo[0] = deviceToUse;
    devInfo[1] = blockSize;
    devInfo[2] = numSMs;
}
