#include <cuda_runtime.h>
#include <cassert>
#include "FixedVectorMath.hh"
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
#define TILE_SIZE 16


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
// ---------------------------------
// 2) Example LN row kernel
// ---------------------------------
__global__ void layerNormRowKernel(
    const float* __restrict__ dIn,
    float* __restrict__ dOut,
    float* __restrict__ dMean,
    float* __restrict__ dVar,
    int rows,
    int cols,
    float epsilon
){
    int row = blockIdx.x; 
    if (row >= rows) return;

    // We'll do naive parallel sums with atomicAdd. 
    __shared__ float sMean;
    __shared__ float sVar;

    if (threadIdx.x == 0) {
        sMean = 0.0f;
        sVar  = 0.0f;
    }
    __syncthreads();

    int startIdx = row*cols;
    float localSum = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        localSum += dIn[startIdx + c];
    }
    atomicAdd(&sMean, localSum);
    __syncthreads();

    if (threadIdx.x == 0) {
        sMean /= (float)cols;
        dMean[row] = sMean;
    }
    __syncthreads();

    float localVar = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float diff = dIn[startIdx + c] - sMean;
        localVar += diff*diff;
    }
    atomicAdd(&sVar, localVar);
    __syncthreads();

    if (threadIdx.x == 0) {
        sVar /= (float)cols;
        dVar[row] = sVar;
    }
    __syncthreads();

    float invStd = rsqrtf(sVar + epsilon);
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float val = dIn[startIdx + c];
        dOut[startIdx + c] = (val - sMean)*invStd;
    }
}

__global__ void dotKernel(
    const float* A, 
    const float* B, 
    float* C,
    int rowsA, 
    int colsA, 
    int colsB
) {
    // Compute row and column for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // If within valid range
    if (row < rowsA && col < colsB) {
        float sum = 0;
        // Multiply row of A by column of B
        for (int k = 0; k < colsA; ++k) {
            sum += A[row * colsA + k] * B[k * colsB + col];
        }
        C[row * colsB + col] = sum;
    }
}

/*
-----------------------------------------
    Cuda API Definitions
-----------------------------------------
*/   
namespace FixedVectorMath {

    void normalizeCuda(
        FixedVector<FixedVector<float>>& matrix,
        FixedVector<float>& means,
        FixedVector<float>& vars,
        float epsilon
    ) {
        int m = matrix.size();
        int n = matrix[0].size();
    
        if ((int)means.size() != m) {
            means = FixedVector<float>(m, 0.0f);
        }
        if ((int)vars.size() != m) {
            vars = FixedVector<float>(m, 0.0f);
        }
    
        // flatten to host
        float* hIn = new float[m*n];
        for (int i = 0; i < m; i++) {
            memcpy(hIn + i*n, matrix[i].data(), n*sizeof(float));
        }
    
        // allocate GPU
        float* dIn=nullptr; 
        float* dOut=nullptr; 
        float* dMean=nullptr; 
        float* dVar=nullptr;
        cudaMalloc(&dIn,  m*n*sizeof(float));
        cudaMalloc(&dOut, m*n*sizeof(float));
        cudaMalloc(&dMean,m*sizeof(float));
        cudaMalloc(&dVar, m*sizeof(float));
    
        // copy input
        cudaMemcpy(dIn, hIn, m*n*sizeof(float), cudaMemcpyHostToDevice);
    
        // launch kernel: 1 block per row, up to 256 threads. 
        dim3 grid(m);
        dim3 block(256);
        layerNormRowKernel<<<grid, block>>>(dIn, dOut, dMean, dVar, m, n, epsilon);
        checkCudaError("layerNormRowKernel");
    
        // copy back
        cudaMemcpy(hIn,   dOut,  m*n*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(means.data(), dMean, m*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(vars.data(),  dVar,  m*sizeof(float), cudaMemcpyDeviceToHost);
    
        // store result
        for (int i = 0; i < m; i++) {
            memcpy(matrix[i].data(), hIn + i*n, n*sizeof(float));
        }
    
        // free
        cudaFree(dIn);
        cudaFree(dOut);
        cudaFree(dMean);
        cudaFree(dVar);
        delete[] hIn;
    }

    FixedVector<FixedVector<float>> dotCuda(
        const FixedVector<FixedVector<float>>& A,
        const FixedVector<FixedVector<float>>& B
    ){
    // Ensure A's columns == B's rows
    int rowsA = A.size();
    int colsA = A[0].size();
    int rowsB = B.size();
    int colsB = B[0].size();

    if (colsA != rowsB) {
        throw std::invalid_argument("Matrix dimension mismatch: A[rows x cols], B[cols x ???]");
    }

    // Flatten A and B for memory transfer to device
    std::vector<float> h_A(rowsA * colsA);
    std::vector<float> h_B(rowsB * colsB);
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsA; ++j) {
            h_A[i * colsA + j] = A[i][j];
        }
    }
    for (int i = 0; i < rowsB; ++i) {
        for (int j = 0; j < colsB; ++j) {
            h_B[i * colsB + j] = B[i][j];
        }
    }

    // Allocate memory on the device
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    size_t sizeA = rowsA * colsA * sizeof(float);
    size_t sizeB = rowsB * colsB * sizeof(float);
    size_t sizeC = rowsA * colsB * sizeof(float);

    cudaMalloc((void**)&d_A, sizeA);
    cudaMalloc((void**)&d_B, sizeB);
    cudaMalloc((void**)&d_C, sizeC);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A.data(), sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), sizeB, cudaMemcpyHostToDevice);

    // Choose reasonable block and grid sizes
    // For simplicity, we set the block to 16x16 threads
    dim3 block(16, 16);
    // We then compute how many blocks we need in each dimension
    dim3 grid((colsB + block.x - 1) / block.x,
              (rowsA + block.y - 1) / block.y);

    // Launch the kernel
    dotKernel<<<grid, block>>>(d_A, d_B, d_C, rowsA, colsA, colsB);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy result back to host
    std::vector<float> h_C(rowsA * colsB);
    cudaMemcpy(h_C.data(), d_C, sizeC, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Reshape flattened C back into 2D vector
    FixedVector<FixedVector<float>> result(rowsA, FixedVector<float>(colsB, static_cast<float>(0)));
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            result[i][j] = h_C[i * colsB + j];
        }
    }

    return result;
    }

    FixedVector<FixedVector<float>> linearCuda(
        FixedVector<FixedVector<float>> A, 
        FixedVector<FixedVector<float>> B,
        FixedVector<float> bias
        ) {
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
            cudaMallocAsync((void**)&dA, m * k * sizeof(float), stream2);
            checkCudaError("CudaMalloc for dA");
            cudaMallocAsync((void**)&dB, k * n * sizeof(float), stream2);
            checkCudaError("CudaMalloc for dB");
            cudaMallocAsync((void**)&dOut, m * n * sizeof(float), stream2);
            checkCudaError("CudaMalloc for dOut");
            cudaMallocAsync((void**)&dBias, n * sizeof(float), stream2);
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

        linear_kernel<<<gridDim, blockDim, m*n*sizeof(float), stream2>>>(m, n, k, dA, dB, dBias, dOut);
        checkCudaError("Linear Kernel Failure");

        //start = std::chrono::high_resolution_clock::now();
        // 4.) Copy the result back to host
        cudaStreamSynchronize(0); // Sync default stram before using dOut on another stream
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

    void addCuda(
        FixedVector<float>& A,
        FixedVector<float>& B
        )
    {
        // 1.) Host Pointers (CPU)
        float* hA = A.data();
        const float* hB = B.data();
        size_t N = A.size();

        // 2.) Allocate GPU memory
        float* dA;
        float* dB;

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

    void addCuda(
        FixedVector<FixedVector<float>>& A, 
        FixedVector<FixedVector<float>>& B
        ) {
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