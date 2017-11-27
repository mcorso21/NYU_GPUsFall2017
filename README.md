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
 *      (2) The occupancy-testing method is user-specified in the program's argument's:
 *
 *          (a) occupancyMethod: 
 *
 *              (1) Blocks per SM: Determines the maximum number of blocks assignable (IE [number of SMs] *
 *              [max blocks assignable to each SM]) and scales it based on the specified targetOccupancy.
 *              The number of threads per block is maxed.
 *
 *              (2) Threads per Block: Determines the maximum number of threads assignable to a block (IE
 *              1024 is common) and scales this based on the specified targetOccupancy. The number of blocks
 *              is equal to ([number of SMs] * [max blocks assignable to each SM]).
 *
 *              (3) Inverted Blocks per Grid to Threads per Block: This combines the previous two tests by
 *              scaling the number of blocks simultaneously assignable (see (1) above) and the max number of
 *              threads per block (see (2) above). These values are inversely scaled based on the specified
 *              targetOccupancy (IE Specifying a 75% occupancy will set Blocks per SM to 75% of capacity and
 *              Threads per Block will be set to 25% of capacity).
 *
 *          (b) targetOccupancy: An integer value of 1 - 100 which specifies the percentage of the maximum 
 *              occupancy for this test.
 *              
 *          (c) problemSize: An integer value which specifies the amount of work to be performed by the kernel
 *              (IE calling with a problemSize of 1000000 will cause the kernel to perform multiplication one
 *              million times)
 *
 *      (3) The work being performed by the threads:
 *      
 *          To avoid variations due to memory, whether they be memory accesses or running out of shared memory
 *          or insufficient registers, we chose to use a simple function which doubles the threads ID and does
 *          not store the result. Each thread will double its thread ID and determine if it needs to perform
 *          additional work. A thread must perform additional work when the problemSize exceeds the number of 
 *          threads in the grid.
 *
 */
