#include <iostream>
#include <curand_kernel.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#define N 1024*1024

__global__ void calculate_pi(int* hits) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Initialize random number state (unique for every thread in the grid)
    int seed = 0;
    int offset = 0;
    curandState_t curand_state;
    curand_init(seed, idx, offset, &curand_state);

    // Generate random coordinates within (0.0, 1.0]
    float x = curand_uniform(&curand_state);
    float y = curand_uniform(&curand_state);

    // Increment hits counter if this point is inside the circle
    if (x * x + y * y <= 1.0f) 
        atomicAdd(hits, 1);
    
}

int main(int argc, char** argv) 
{
    // Initialize NVSHMEM
    nvshmem_init();

    // Obtain our NVSHMEM processing element ID
    int my_pe = nvshmem_my_pe();

    // Each PE (arbitrarily) chooses the GPU corresponding to its ID
    int device = my_pe;
    cudaSetDevice(device);

    // Allocate host and device values
    int* hits;
    hits = (int*) malloc(sizeof(int));

    int* d_hits;
    cudaMalloc((void**) &d_hits, sizeof(int));

    // Initialize number of hits and copy to device
    *hits = 0;
    cudaMemcpy(d_hits, hits, sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel to do the calculation
    int threads_per_block = 256;
    int blocks = (N + threads_per_block - 1) / threads_per_block;

    calculate_pi<<<blocks, threads_per_block>>>(d_hits);
    cudaDeviceSynchronize();

    // Copy final result back to the host
    cudaMemcpy(hits, d_hits, sizeof(int), cudaMemcpyDeviceToHost);

    // Calculate final value of pi
    float pi_est = (float) *hits / (float) (N) * 4.0f;

    // Print out result
    std::cout << "Estimated value of pi on PE " << my_pe << " = " << pi_est << std::endl;
    std::cout << "Relative error on PE " << my_pe << " = " << std::abs((M_PI - pi_est) / pi_est) << std::endl;

    free(hits);
    cudaFree(d_hits);

    // Finalize nvshmem
    nvshmem_finalize();

    return 0;
}