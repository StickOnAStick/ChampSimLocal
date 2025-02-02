#include <cuda_runtime.h>
#include <cassert>
#include "FixedVector.hh"
#include <cmath>
#include <chrono>
#include <iostream>
#include <cudaProfiler.h>
#include <cstring>

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
    // Calculate the row and column index for this thread
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Make sure we don't go out of bounds
    if (row < M && col < N) {
        float tmp = 0.0f;
        
        // Perform the dot product of the row from A and column from B
        // Probably could could sgemm but im lazy 
        for (int i = 0; i < K; ++i) 
            tmp += A[row * K + i] * B[i * N + col];

        // Add the bias term
        C[row * N + col] += bias[col];
    }
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




/*
-----------------------------------------
    Cuda API Definitions
-----------------------------------------
*/   
namespace CudaFixedVectorMath {
    FixedVector<FixedVector<float>> dotproduct(FixedVector<FixedVector<float>>& A, FixedVector<FixedVector<float>>& B) {
        
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
            if (dA != nullptr) cudaFree(dA);
            if (dB != nullptr) cudaFree(dB);
            if (dOut != nullptr) cudaFree(dOut);

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
        cudaMemcpyAsync(dA, hA, m * n * sizeof(float), cudaMemcpyHostToDevice);
        checkCudaError("cudaMemcpyAsync for dA");
        cudaMemcpyAsync(dB, hB, k * n * sizeof(float), cudaMemcpyHostToDevice);
        checkCudaError("cudaMemcpyAsync for dB");
        cudaMemsetAsync(dOut, 0, m * k * sizeof(float));
        checkCudaError("cudaMemsetAsync for dOut");

        // Launch the kernel
        dim3 blockDim(16, 16);
        dim3 gridDim((m + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);  // Grid size based on matrix dimensions
        sgemm_naive<<<gridDim, blockDim>>>(m, n, k, dA, dB, dOut);
        checkCudaError("Kernel launch failed");

        // Copy result back to host asynchronously
        cudaMemcpyAsync(hOut, dOut, m * k * sizeof(float), cudaMemcpyDeviceToHost);
        checkCudaError("cudaMemcpyAsync for hOut");

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
        if (A.size() != B.size() || A.size() != out.size())
            throw std::invalid_argument("Size mismatch between out = A*B matricies.");
        
        int m = A.size();     // Number of rows in A
        int n = A[0].size();  // Number of columns in A

        // Flatten the 2D vectors into contiguous 1D arrays
        float* hA = new float[m * n];
        float* hB = new float[m * n];
        float* hOut = new float[m * n];
        auto start = std::chrono::high_resolution_clock::now();
       
        for (size_t i = 0; i < m; ++i) {
            std::memcpy(hA + i * n, A[i].data(), n * sizeof(float));
            std::memcpy(hB + i * n, B[i].data(), n * sizeof(float));
        }
        auto finish = std::chrono::high_resolution_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count() << " ns for memcpy\n";

        // Device memory allocations (only if dimensions have changed)
        static float* dA = nullptr;
        static float* dB = nullptr;
        static float* dOut = nullptr;
        static int prevM = -1, prevN = -1;

        // Check if the previous dimensions were the same, if not we need to free the pointers on the device
        if (prevM != m || prevN != n) {
            if (dA != nullptr) cudaFree(dA);
            if (dB != nullptr) cudaFree(dB);
            if (dOut != nullptr) cudaFree(dOut);

            // Allocate device memory
            cudaMallocAsync((void**)&dA, m * n * sizeof(float),0);
            checkCudaError("cudaMalloc for dA");
            cudaMallocAsync((void**)&dB, m * n * sizeof(float),0);
            checkCudaError("cudaMalloc for dB");
            cudaMallocAsync((void**)&dOut, m * n * sizeof(float),0);
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
        mul2DKernel<<<gridDim, blockDim>>>(dA, dB, dOut, m, n);
        checkCudaError("Kernel launch failed");

        
        // Copy result back to host asynchronously
        cudaMemcpyAsync(hOut, dOut, m * n * sizeof(float), cudaMemcpyDeviceToHost);
        checkCudaError("cudaMemcpyAsync for hOut");
        
        for (size_t i = 0; i < m; ++i) 
            memcpy(out[i].data(), hOut + i * n, n * sizeof(float));  // Copy row i of the result
        
        delete[] hA;
        delete[] hB;
        delete[] hOut;
    }

    FixedVector<FixedVector<float>> linear(FixedVector<FixedVector<float>> A, 
                                           FixedVector<FixedVector<float>> B, 
                                           FixedVector<float> bias) {
        //auto start = std::chrono::high_resolution_clock::now();
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
        cudaMemsetAsync(dOut, 0, m * n * sizeof(float));
        checkCudaError("cudaMemsetAsync for dOut");

        // 3.) Launch Kernel
        dim3 blockDim(8, 8);  // Smaller block size to reduce resource usage
        dim3 gridDim((m + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);  // Grid size based on matrix dimensions

        linear_kernel<<<gridDim, blockDim>>>(m, n, k, dA, dB, dBias, dOut);
        checkCudaError("Linear Kernel Failure");
        //start = std::chrono::high_resolution_clock::now();
        // 4.) Copy the result back to host
        cudaMemcpyAsync(hOut, dOut, m * n * sizeof(float), cudaMemcpyDeviceToHost);
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