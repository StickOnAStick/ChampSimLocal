#include <cuda_runtime.h>
#include <cassert>
#include "FixedVector.hh"
#include <cmath>
#include <chrono>
#include <iostream>
#include <cudaProfiler.h>
#include <cstring>
#include <omp.h>

void checkCudaError(const char* message) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << message << " - " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
}
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))



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
__global__ void mulKernel(const float* A, const float* B, float* out, size_t N){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N){
        out[idx] = A[idx] * B[idx];
    }
}

__global__ void mul2DKernel(const float* A, const float* B, float* out, size_t width, size_t height){
    //1. Determine row and column for the current thread
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    // 2. Check Boundary 
    if (row < height && col < width){
        size_t idx = row * width + col;
        out[idx] = A[idx] * B[idx];
    }
}

__global__ void addKernel(float* A, const float* B, size_t N){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N){
        A[idx] += B[idx];
    }
}

__global__ void add2DKernel(float* A, const float* B, size_t width, size_t height){
    //1. Determine row and column for the current thread
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    // 2. Check Boundary 
    if (row < height && col < width){
        size_t idx = row * width + col;
        A[idx] += B[idx];
    }
}

__global__ void linear_kernel(int M, int N, int K, const float *A, const float *B, const float *bias, float *C) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    // keep within tile bounds 
    if (x < M && y < N) {
      float tmp = 0.0;
      for (int i = 0; i < K; ++i) 
        tmp += A[x * K + i] * B[i * N + y];
      C[x * N + y] =  tmp + C[x * N + y];
    }
    C[x * N + y] += bias[y];
}

__global__ void sgemm_naive(int M, int N, int K, const float *A,
                            const float *B, float *C) {
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;
  // keep within tile bounds 
  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) 
      tmp += A[x * K + i] * B[i * N + y];
    C[x * N + y] =  tmp + C[x * N + y];
  }
}

__global__ void elementWiseMultiplyKernel(float* dA, float* dB, float* dOut, int m, int n)
{
    // Calculate global thread index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Perform the element-wise multiplication if within bounds
    if (row < m && col < n)
    {
        int index = row * n + col;
        dOut[index] = dA[index] * dB[index];
    }
}

/*
-----------------------------------------
    Cuda API Definitions
-----------------------------------------
*/   
namespace CudaFixedVectorMath {
    FixedVector<FixedVector<float>> dotproduct(FixedVector<FixedVector<float>>& A, FixedVector<FixedVector<float>>& B) {
        
        static cudaStream_t stream1;
        // std::cout << stream1;
        if (stream1 == 0)
            cudaStreamCreate (&stream1);
        int m = A.size();  // Number of rows in A
        int n = A[0].size();  // Number of columns in A (also number of rows in B)
        int k = B[0].size();  // Number of columns in B

        // Flatten the 2D vectors into contiguous 1D arrays
        float* hA = new float[m * n];
        float* hB = new float[n * k];
        float* hOut = new float[m * k];

        for (size_t i = 0; i < m; ++i) {
            memcpy(hA + i * n, A[i].data(), n * sizeof(float));
        }
        for (size_t i = 0; i < n; ++i) {
            memcpy(hB + i * k, B[i].data(), k * sizeof(float));
        }

        // Device memory allocations (only if dimensions have changed)
        static float* dA = nullptr;
        static float* dB = nullptr;
        static float* dOut = nullptr;

        static int prevM = -1, prevN = -1, prevK = -1;

        // Check if the previous dimensions were the same, if not we need to 
        // free the pointers on the device
        if (prevM != m || prevK != k || prevN != n) {
            printf("resize");
            cudaFreeAsync(dA,0);
            cudaFreeAsync(dB,0);
            cudaFreeAsync(dOut,0);

            // Allocate device memory
            cudaMallocAsync((void**)&dA, m * n * sizeof(float),0);
            checkCudaError("cudaMalloc for dA");
            cudaMallocAsync((void**)&dB, n * k * sizeof(float),0);
            checkCudaError("cudaMalloc for dB");
            cudaMallocAsync((void**)&dOut, m * k * sizeof(float),0);
            checkCudaError("cudaMalloc for dOut");

            prevM = m;
            prevK = k;
            prevN = n;
        }

        // Copy A and B to device memory asynchronously
        //auto start = std::chrono::high_resolution_clock::now();
        cudaMemcpyAsync(dA, hA, m * n * sizeof(float), cudaMemcpyHostToDevice,0);
        checkCudaError("cudaMemcpyAsync for dA");
        //auto finish = std::chrono::high_resolution_clock::now();
        //std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count() << " ns for memcpy1 \n";
        //start = std::chrono::high_resolution_clock::now();
        cudaMemcpyAsync(dB, hB, k * n * sizeof(float), cudaMemcpyHostToDevice,0);
        checkCudaError("cudaMemcpyAsync for dB");
        // finish = std::chrono::high_resolution_clock::now();
        //std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count() << " ns for memcpy2 \n";
        cudaMemsetAsync(dOut, 0, m * k * sizeof(float),0);
        checkCudaError("cudaMemsetAsync for dOut");

        // Launch the kernel
        dim3 blockDim(16, 16);
        dim3 gridDim((m + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);  // Grid size based on matrix dimensions
        sgemm_naive<<<gridDim, blockDim>>>(m, n, k, dA, dB, dOut);
        checkCudaError("Kernel launch failed");
        
        
     
        // Copy result back to host asynchronously
       

        //start = std::chrono::high_resolution_clock::now();

        cudaMemcpyAsync(hOut, dOut, m * k * sizeof(float), cudaMemcpyDeviceToHost,stream1);
        checkCudaError("cudaMemcpyAsync for hOut");
        //finish = std::chrono::high_resolution_clock::now();
        //std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count() << " ns for memcpy3 \n";

        // Convert output back to FixedVector
        
        FixedVector<FixedVector<float>> out(m, FixedVector<float>(k, 0.0f));
        for (size_t i = 0; i < m; ++i) {
            memcpy(out[i].data(), hOut + i * k, k * sizeof(float));  // Copy row i of the result
        }

        
        // Cleanup (free memory)
        delete[] hA;
        delete[] hB;
        delete[] hOut;

        return out;
    }

    void mul(FixedVector<float>& out, FixedVector<float>& A, FixedVector<float>& B)
    {
        assert(A.size() == B.size());
        size_t N = A.size();

        // 1.) Host pointers (CPU)
        const float*  hA   = A.data();
        const float*  hB   = B.data();
        float*  hOut = out.data();
        
        // 2.) Device Pointers (GPU)
        float* dA   = nullptr;  
        float* dB   = nullptr;
        float* dOut = nullptr;

        cudaMallocAsync(&dA, N * sizeof(float),0);
        cudaMallocAsync(&dB, N * sizeof(float),0);
        cudaMallocAsync(&dOut, N * sizeof(float),0);

        // 3.) Copy input from host to device
        cudaMemcpy(dA, hA, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB, N * sizeof(float), cudaMemcpyHostToDevice);

        // 4.) Launch Kernel
        int blockSize = 256; // This is mostly arbitrary! But we use 16 for a base 2 number less than 1024 that can still perform many operations.
        int gridSize = (N + blockSize - 1) / blockSize; 
        mulKernel<<<gridSize, blockSize>>>(dA, dB, dOut, N);
        // 5.) Copy the output from device to host
        // BEWARE: using vec.data() as a the T* works for all values EXCEPT BOOLS, as vectors of bools are NOT CONTIGUOUS.
        cudaMemcpy(hOut,dOut, N * sizeof(float), cudaMemcpyDeviceToHost);

        // 6.) Clean up
    }

void mul(FixedVector<FixedVector<float>>& out, FixedVector<FixedVector<float>>& A, FixedVector<FixedVector<float>>& B)
{
    int m = A.size();  // Number of rows
    int n = A[0].size();  // Number of columns

    // Allocate device memory for matrices A, B, and out
    float* dA = nullptr;
    float* dB = nullptr;
    float* dOut = nullptr;

    // Allocate contiguous memory for all matrices (A, B, and out)
    cudaMallocAsync((void**)&dA, m * n * sizeof(float), 0);
    checkCudaError("cudaMalloc for dA");

    cudaMallocAsync((void**)&dB, m * n * sizeof(float), 0);
    checkCudaError("cudaMalloc for dB");

    cudaMallocAsync((void**)&dOut, m * n * sizeof(float), 0);
    checkCudaError("cudaMalloc for dOut");

    // Flatten the matrices and copy to device (single memory copy)
    float* hA = new float[m * n];
    float* hB = new float[m * n];


    // Flatten the 2D matrix A to 1D array
    for (int i = 0; i < m; ++i) {
        std::memcpy(hA + i * n, A[i].data(), n * sizeof(float));
        std::memcpy(hB + i * n, B[i].data(), n * sizeof(float));
    }
    
    // Copy the flattened matrices to device
    cudaMemcpyAsync(dA, hA, m * n * sizeof(float), cudaMemcpyHostToDevice);
    checkCudaError("cudaMemcpy for dA");

    cudaMemcpyAsync(dB, hB, m * n * sizeof(float), cudaMemcpyHostToDevice);
    checkCudaError("cudaMemcpy for dB");

    // Set up kernel launch parameters (using 16x16 block size)
    dim3 blockDim(16, 16);  // Block size: 16x16 threads
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (m + blockDim.y - 1) / blockDim.y);  // Grid size

    // Launch the kernel for element-wise multiplication
    elementWiseMultiplyKernel<<<gridDim, blockDim>>>(dA, dB, dOut, m, n);
    checkCudaError("Kernel launch failed");

    // Copy the result back to the host in one go (flattened)
    float* hOut = new float[m * n];
    cudaMemcpyAsync(hOut, dOut, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    checkCudaError("cudaMemcpy for hOut");

    // Copy the flattened result back into the 2D out structure
    for (int i = 0; i < m; ++i) {
        std::memcpy(out[i].data(), hOut + i * n, n * sizeof(float));
    }

    // Free host memory
    delete[] hA;
    delete[] hB;
    delete[] hOut;
}

    FixedVector<FixedVector<float>> linear(FixedVector<FixedVector<float>> A, 
                                           FixedVector<FixedVector<float>> B, 
                                           FixedVector<float> bias) {
        static cudaStream_t stream2;
        // std::cout << stream1;
        if (stream2 == 0)
            cudaStreamCreate (&stream2);
        int m = A.size();    // Number of rows in A
        int n = B[0].size(); // Number of columns in B
        int k = A[0].size(); // Number of columns in A (also the number of rows in B)

        
        // 1.) Flatten the 2D vectors into contiguous 1D arrays
        float* hA = new float[m * k];
        float* hB = new float[k * n];
        float* hBias = new float[n];
        float* hOut = new float[m * n];

        // Copy the values into the flattened arrays
        for (size_t i = 0; i < m; ++i) {
            memcpy(hA + i * k, A[i].data(), k * sizeof(float));  // Flatten row i of A
        }

        for (size_t i = 0; i < k; ++i) {
            memcpy(hB + i * n, B[i].data(), n * sizeof(float));  // Flatten row i of B
        }

        memcpy(hBias, bias.data(), n * sizeof(float));  // Copy the bias values
    

        // Allocate memory on the GPU only once
        static float* dA = nullptr;
        static float* dB = nullptr;
        static float* dBias = nullptr;
        static float* dOut = nullptr;
        static int prevM = -1, prevN = -1, prevK = -1;  // Track previous matrix dimensions
        // Check if the size has changed and reallocate if necessary
        if (prevM != m || prevK != k || prevN != n) {
            // If dimensions are different, free the previous memory and allocate new memory
            if (dA != nullptr) cudaFree(dA);
            if (dB != nullptr) cudaFree(dB);
            if (dOut != nullptr) cudaFree(dOut);
            if (dBias != nullptr) cudaFree(dBias);
            // Allocate memory for the new matrices on the device
            cudaMallocAsync((void**)&dA, m * k * sizeof(float),0);
            checkCudaError("CudaMalloc for dA");
            cudaMallocAsync((void**)&dB, k * n * sizeof(float),0);
            checkCudaError("CudaMalloc for dB");
            cudaMallocAsync((void**)&dOut, m * n * sizeof(float),0);
            checkCudaError("CudaMalloc for dOut");
            cudaMallocAsync((void**)&dBias, n * sizeof(float),0);
            checkCudaError("CudaMalloc for dBias");

            // Update the previous dimensions
            prevM = m;
            prevK = k;
            prevN = n;
        }


        // 2.) Copy data from host to device
        cudaMemcpyAsync(dA, hA, m * k * sizeof(float), cudaMemcpyHostToDevice);
        checkCudaError("cudaMemcpyAsync for dA");
        cudaMemcpyAsync(dB, hB, k * n * sizeof(float), cudaMemcpyHostToDevice);
        checkCudaError("cudaMemcpyAsync for dB");
        cudaMemcpyAsync(dBias, hBias, n * sizeof(float), cudaMemcpyHostToDevice);
        checkCudaError("cudaMemcpyAsync for dBias");
        cudaMemsetAsync(dOut, 0, m * n * sizeof(float),0);
        checkCudaError("cudaMemsetAsync for dOut");

        // 3.) Launch Kernel
        dim3 blockDim(8, 8);  // Smaller block size to reduce resource usage
        dim3 gridDim((m + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);  // Grid size based on matrix dimensions

        linear_kernel<<<gridDim, blockDim>>>(m, n, k, dA, dB, dBias, dOut);
        checkCudaError("Linear Kernel Failure");
        //start = std::chrono::high_resolution_clock::now();
        // 4.) Copy the result back to host
        cudaMemcpyAsync(hOut, dOut, m * n * sizeof(float), cudaMemcpyDeviceToHost,stream2);
        checkCudaError("cudaMemcpyAsync for Hout");

        // 5.) Convert the result back to a FixedVector
        FixedVector<FixedVector<float>> result(m, FixedVector<float>(n));
        for (size_t i = 0; i < m; ++i) {
            memcpy(result[i].data(), hOut + i * n, n * sizeof(float));  // Copy each row to the result
        }

        // 6.) Clean up
        delete[] hA;
        delete[] hB;
        delete[] hBias;
        delete[] hOut;
        return result;
    }

    void add(FixedVector<float>& A, FixedVector<float>& B)
    {
        // 1.) Host Pointers (CPU)
        float* hA = A.data();
        const float* hB = B.data();
        size_t N = A.size();

        // 2.) Allocate GPU memory
        float* dA = new float[N];
        float* dB = new float[N];

        cudaMalloc(&dA, N * sizeof(float));
        cudaMalloc(&dB, N * sizeof(float));

        // 3.) Copy input to GPU
        cudaMemcpy(dA, hA, N * sizeof(float), cudaMemcpyHostToDevice);
        checkCudaError("cudaMalloc for dA");
        cudaMemcpy(dB, hB, N * sizeof(float), cudaMemcpyHostToDevice);
        checkCudaError("cudaMalloc for dB");

        // 4.) Launch Kernel
        int blockSize = 256; // Still arbitrary...
        int gridSize = (N + blockSize - 1) / blockSize;
        addKernel<<<gridSize, blockSize>>>(dA, dB, N);
        checkCudaError("Kernel launch failed");

        // 5.) Copy device output to host
        cudaMemcpy(hA, dA, N * sizeof(float), cudaMemcpyDeviceToHost);
        checkCudaError("cudaMemcpy for hOut");

        // 6.) Everybody clean up
        delete[] hA;
        delete[] hB;
        cudaFree(dA);
        cudaFree(dB);
    }


    void add(FixedVector<FixedVector<float>>& A, FixedVector<FixedVector<float>>& B) {
        int m = A.size();     // Number of rows in A
        int n = A[0].size();  // Number of columns in A
        
        // Flatten the 2D vectors into contiguous 1D arrays
        float* hA = new float[m * n];
        float* hB = new float[m * n];

        for (size_t i = 0; i < m; ++i){
            memcpy(hA + i * n, A[i].data(), n * sizeof(float));
            memcpy(hB + i * n, B[i].data(), n * sizeof(float));
        }

        // Device memory allocations (only if dimensions have changed)
        static float* dA = nullptr;
        static float* dB = nullptr;
        static int prevM = -1, prevN = -1;

        // Check if the previous dimensions were the same, if not we need to 
        // free the pointers on the device
        if (prevM != m || prevN != n) {
            if (dA != nullptr) cudaFree(dA);
            if (dB != nullptr) cudaFree(dB);

            // Allocate device memory
            cudaMallocAsync((void**)&dA, m * n * sizeof(float),0);
            checkCudaError("cudaMalloc for dA");
            cudaMallocAsync((void**)&dB, m * n * sizeof(float),0);
            checkCudaError("cudaMalloc for dB");

            prevM = m;
            prevN = n;
        }

        // Copy A and B to device memory asynchronously
        cudaMemcpyAsync(dA, hA, m * n * sizeof(float), cudaMemcpyHostToDevice);
        checkCudaError("cudaMemcpyAsync for dA");
        cudaMemcpyAsync(dB, hB, m * n * sizeof(float), cudaMemcpyHostToDevice);
        checkCudaError("cudaMemcpyAsync for dB");

        // Launch the kernel
        dim3 blockDim(16, 16);
        dim3 gridDim((m + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);  // Grid size based on matrix dimensions
        add2DKernel<<<gridDim, blockDim>>>(dA, dB, m, n);
        checkCudaError("Kernel launch failed");
        
        // Copy result back to host asynchronously
        cudaMemcpyAsync(hA, dA, m * n * sizeof(float), cudaMemcpyDeviceToHost);
        checkCudaError("cudaMemcpyAsync for hOut");
        
        for (size_t i = 0; i < m; ++i) {
                memcpy(A[i].data(), hA + i * n, n * sizeof(float));  // Copy row i of the result
        }
        delete[] hA;
        delete[] hB;
    }
}