#include <cuda_runtime.h>
#include <cassert>
#include "FixedVectorMath.hh"


/*
    This file is split into two sections!

    1. Kernel code
        - This is the code executed on the GPU itself
    2. Cuda Definitions
        - Our API to interact with the GPU, responsible for data preperation, transmission, and return

    ////////////////////////////////////////
    // Important Variables
    ////////////////////////////////////////

    blockIdx: Which block (in a given dimension denoted by .(x,y,z)) the current thread belongs to.

    blockDim: How many threads per block along the given axis .(x, y, z)
        - Nvidia executes threads in groups of 32 called warps. It's optimal (not required) to keep block sizes a multiple of 32


    Refer to the Cheat Sheet for visualizing these dimensions. 
    https://www.eecs.umich.edu/courses/eecs471/resources/materials/CUDA-Thread-Indexing-Cheatsheet.pdf
*/

/*
-----------------------------------------
    Kernel function defintions
-----------------------------------------
*/
template <typename T>
__global__ void mulKernel(const T* A, const T* B, T* out, size_t N){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N){
        out[idx] = A[idx] * B[idx];
    }
}

template <typename T>
__global__ void mul2DKernel(const T* A, const T* B, T* out, size_t width, size_t height){
    //1. Determine row and column for the current thread
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    // 2. Check Boundary 
    if (row < height && col < width){
        size_t idx = row * width + col;

        out[idx] = A[idx] * B[idx];
    }
}

template <typename T>
__global__ void addKernel(const T* A, const T* B, T* out, size_t N){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N){
        out[idx] = A[idx] + B[idx];
    }
}

template <typename T>
__global__ void add2DKernel(const T* A, const T* B, T* out, size_t width, size_t height){
    //1. Determine row and column for the current thread
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    // 2. Check Boundary 
    if (row < height && col < width){
        size_t idx = row * width + col;

        out[idx] = A[idx] + B[idx];
    }
}

/*
-----------------------------------------
    Cuda API Definitions
-----------------------------------------
*/
template <typename T>
void mulCuda(FixedVector<T>& out, const FixedVector<T>& A, const FixedVector<T>& B){
    assert(A.size() == B.size());
    assert(out.size() == A.size());
    size_t N = A.size();

    // 1.) Host pointers (CPU)
    const T* hA   = A.data();
    const T* hB   = B.data();
    T*       hOut = out.data();

    // 2.) Device Pointers (GPU)
    T* dA   = nullptr;  
    T* dB   = nullptr;
    T* dOut = nullptr;

    cudaMalloc(&dA, N * sizeof(T));
    cudaMalloc(&dB, N * sizeof(T));
    cudaMalloc(&dOut, N * sizeof(T));

    // 3.) Copy input from host to device
    cudaMemcpy(dA, hA, N * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, N * sizeof(T), cudaMemcpyHostToDevice);

    // 4.) Launch Kernel
    int blockSize = 16; // This is mostly arbitrary! But we use 16 for a base 2 number less than 1024 that can still perform many operations.
    int gridSize = (N + blockSize - 1) / blockSize; 
    mulKernel<T><<<gridSize, blockSize>>>(dA, dB, dOut, N);

    // 5.) Copy the output from device to host
    cudaMemcpy(hOut, dOut, N*sizeof(T), cudaMemcpyDeviceToHost);

    // 6.) Clean up
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dOut);

}