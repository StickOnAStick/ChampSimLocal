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
#define TILE_SIZE 32
#define BLOCK_SIZE 32

void print_attention_scores_per_head(float* d_attention_scores, int num_heads, int sequence_len) {
    float* h_attention_scores = new float[num_heads * sequence_len * sequence_len];
    cudaMemcpy(h_attention_scores, d_attention_scores, num_heads * sequence_len * sequence_len * sizeof(float), cudaMemcpyDeviceToHost);

    for (int head = 0; head < num_heads; ++head) {
        std::cout << "Head " << head << " Attention Scores:\n";
        for (int i = 0; i < sequence_len; ++i) {
            for (int j = 0; j < sequence_len; ++j) {
                std::cout << h_attention_scores[head * sequence_len * sequence_len + i * sequence_len + j] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    delete[] h_attention_scores;
}

void print_device_matrix(const float* d_matrix, int rows, int cols, const std::string& name, cudaStream_t stream) {
    // Allocate host memory
    float* h_matrix = new float[rows * cols];

    // Copy from device to host
    cudaMemcpyAsync(h_matrix, d_matrix, rows * cols * sizeof(float), cudaMemcpyDeviceToHost,stream);
    cudaDeviceSynchronize();  // Ensure memory copy is completed before printing

    // Print the first 5x5 submatrix
    std::cout << "First 5x5 submatrix of " << name << ":\n";
    for (int i = 0; i < 24 && i < rows; ++i) {  // Ensure within bounds
        std::cout << i << ":";
        for (int j = 0; j < 70 && j < cols; ++j) {
            std::cout << h_matrix[i * cols + j] << " ";
        }
        std::cout << "\n";
    }

    // Free host memory
    delete[] h_matrix;
}

void printmatrix(const FixedVector<FixedVector<float>>& matrix, const std::string& name) {
    std::cout << "Matrix: " << name << std::endl;
    for (size_t i = 0; i < std::min(matrix.size(), static_cast<size_t>(5)); ++i) {
        for (size_t j = 0; j < std::min(matrix[i].size(), static_cast<size_t>(5)); ++j) {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
  }
  
  void print_dmask(float* dmask, int m) {
    // Allocate host memory
    float* hmask = new float[m * m];

    // Copy from device to host
    cudaMemcpy(hmask, dmask, m * m * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the first 5x5 submatrix
    std::cout << "First 5x5 submatrix of dmask:\n";
    for (int i = 0; i < 5 && i < m; ++i) {  // Ensure within bounds
        for (int j = 0; j < 5 && j < m; ++j) {
            std::cout << hmask[i * m + j] << " ";
        }
        std::cout << "\n";
    }

    // Free host memory
    delete[] hmask;
}

  void print_hmask(const float* hmask, int m) {
    std::cout << "First 5x5 submatrix of hmask:\n";
    for (int i = 0; i < 5 && i < m; ++i) {  // Ensure within bounds
        for (int j = 0; j < 5 && j < m; ++j) {
            std::cout << hmask[i * m + j] << " ";
        }
        std::cout << "\n";
    }
}
void printMatrix(const char* name, float* d_matrix, int rows, int cols) {
    FixedVector<float> h_matrix(rows * cols);
    cudaMemcpy(h_matrix.data(), d_matrix, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << name << " (First 5x5 values):\n";
    int display_rows = std::min(5, rows);
    int display_cols = std::min(5, cols);
    
    for (int i = 0; i < display_rows; ++i) {
        for (int j = 0; j < display_cols; ++j) {
            std::cout << h_matrix[i * cols + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "--------------------------------------\n";
}
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
#define TILE_SIZE 32
__global__ void sgemm_blockheads(int m, int n, int k, 
    const float* sequence_history, const float *w_q, const float *w_v, const float *w_k,
    float* Q, float* V, float* K) {

    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_Bq[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_Bv[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_Bk[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sumQ = 0.0f, sumV = 0.0f, sumK = 0.0f;

    for (int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load elements into shared memory
        if (row < m && t * TILE_SIZE + tx < k) {
            tile_A[ty][tx] = sequence_history[row * k + t * TILE_SIZE + tx];
        } else {
            tile_A[ty][tx] = 0.0f;
        }

        if (t * TILE_SIZE + ty < k && col < n) {
            tile_Bq[ty][tx] = w_q[(t * TILE_SIZE + ty) * n + col];
            tile_Bv[ty][tx] = w_v[(t * TILE_SIZE + ty) * n + col];
            tile_Bk[ty][tx] = w_k[(t * TILE_SIZE + ty) * n + col];
        } else {
            tile_Bq[ty][tx] = 0.0f;
            tile_Bv[ty][tx] = 0.0f;
            tile_Bk[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial sum within tile
        for (int i = 0; i < TILE_SIZE; ++i) {
            sumQ += tile_A[ty][i] * tile_Bq[i][tx];
            sumV += tile_A[ty][i] * tile_Bv[i][tx];
            sumK += tile_A[ty][i] * tile_Bk[i][tx];
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        Q[row * n + col] += sumQ;
        V[row * n + col] += sumV;
        K[row * n + col] += sumK;
    }
}

__global__ void compute_attention_scores(
    const float* Q, const float* K, const float* mask,
    float* attention_scores, int sequence_len, int d_head,
    bool use_mask, int num_heads, int d_model) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int head = blockIdx.z;  // Each block in the z-dimension corresponds to a different head

    if (row >= sequence_len || col >= sequence_len || head >= num_heads) return;

    int head_offset = head * d_head;
    float score = 0.0f;

    // Compute raw attention scores (QK^T)
    for (int k = 0; k < d_head; ++k) {
        int Q_index = row * d_model + head_offset + k;
        int K_index = col * d_model + head_offset + k;
        score += Q[Q_index] * K[K_index];
    }

    // Scale the score
    score /= sqrtf(static_cast<float>(d_head));

    // Apply mask if necessary
    if (use_mask) {
        int mask_index = row * sequence_len + col;
        score += mask[mask_index];  // Ensure mask is correctly indexed
    }

    // Store attention score for this head
    int output_index = head * sequence_len * sequence_len + row * sequence_len + col;
    attention_scores[output_index] = score;
}


__global__ void compute_head_out(
    float* output, const float* att_score_soft, const float* V, 
    const int sequence_len, const int d_model, const int head, const int d_head) 
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    int head_offset = head * d_head;

    if (row < sequence_len && col < d_head) {
        float tmp = 0.0f;
        for (int i = 0; i < sequence_len; ++i) {
            tmp += att_score_soft[row * sequence_len + i] * V[i * d_model + head_offset + col];
        }
        output[row * d_head + col] = tmp;
    }
}


__global__ void merge_heads(const float* head_out,  float* attention_out, int sequence_len, int d_model, int num_heads, int d_head, int head) // Head index passed as input
{
    int seq_idx = blockIdx.x * blockDim.x + threadIdx.x; // Row index (sequence)
    int feature_idx = blockIdx.y * blockDim.y + threadIdx.y; // Column index (within d_head)

    if (seq_idx < sequence_len && feature_idx < d_head) {
        // Compute correct input index in head_out for this head
        int head_out_index = seq_idx * d_head + feature_idx;

        // Compute correct output index in attention_out
        int attention_out_index = seq_idx * d_model + head * d_head + feature_idx;
        __syncthreads();
        attention_out[attention_out_index] = head_out[head_out_index];
    }
}


__global__ void softMax(float* output, float* input, int M, int N) 
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < M) {
        // maximum of this row
        float x_max = -INFINITY;
        // norm factor of this row
        float norm = 0.0f;

        // output in 3 passes
        for (int col = 0; col < N; col++) {
            int i = row * N + col;
            x_max = max(x_max, input[i]);
        }
        for (int col = 0; col < N; col++) {
            int i = row * N + col;
            norm += expf(input[i] - x_max);
        }
        for (int col = 0; col < N; col++) {
            int i = row * N + col;
            output[i] = expf(input[i] - x_max) / norm;
        }
    }
}

#define BLOCK_SIZE 32
/*
-----------------------------------------
    Cuda API Definitions
-----------------------------------------
*/   
namespace FixedVectorMath {

    FixedVector<FixedVector<float>> MMA_CUDA(
        bool use_mask,
        int sequence_len,
        int d_model,
        const int num_ma_heads, 
        FixedVector<FixedVector<float>> &sequence_history,
        FixedVector<FixedVector<float>> &w_q,
        FixedVector<FixedVector<float>> &w_k,
        FixedVector<FixedVector<float>> &w_v,
        FixedVector<FixedVector<float>> &w_o) {
         
        cudaStream_t streams[num_ma_heads];
        for (int i = 0; i < num_ma_heads; i++)
            cudaStreamCreate(&streams[i]);
    
        FixedVector<FixedVector<float>> attention_out(sequence_len, FixedVector<float>(d_model, 0.0f));
        FixedVector<FixedVector<float>> mask(sequence_len, FixedVector<float>(sequence_len, 0.0f));
       
        if (use_mask)
          FixedVectorMath::applyMask(mask);
        

        // Fix dimensions
        int m = sequence_len;  // Number of sequences
        int n = d_model;       // Model dimension
        int k = d_model;       // Embedding dimension
        int d_head = d_model / num_ma_heads;
        int num_heads = num_ma_heads;

        // Allocate host memory
        float* hsequence_history = new float[m * k];
        float* hmask = new float[m * m];  // Flattened mask
        float* hout = new float[m * d_model];  
        float* hw_q = new float[k * d_model];
        float* hw_v = new float[k * d_model];
        float* hw_k = new float[k * d_model];
        
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < m; ++j) {
                hmask[i * m + j] = mask[i][j];  // Flattening row-major order
            }
        }
       
        // Copy input matrices to 1D arrays
        for (size_t i = 0; i < m; ++i) {
            memcpy(hsequence_history + i * k, sequence_history[i].data(), k * sizeof(float));
        }
        for (size_t i = 0; i < k; ++i) {
            memcpy(hw_q + i * d_model, w_q[i].data(), d_model * sizeof(float));
            memcpy(hw_v + i * d_model, w_v[i].data(), d_model * sizeof(float));
            memcpy(hw_k + i * d_model, w_k[i].data(), d_model * sizeof(float));
        }
    
       // Device memory
        static float* dsequence_history = nullptr;
        static float* dattention_scores = nullptr;
        static float* dattention_scores_softmax = nullptr;
        static float* dOut = nullptr;
        static float* dw_q = nullptr;
        static float* dmask = nullptr;
        static float* dw_k = nullptr;
        static float* dw_v = nullptr;
        static float* dQ = nullptr;
        static float* dV = nullptr;
        static float* dK = nullptr;
        static float* dhead_out = nullptr;
        static float* dattention_out = nullptr;

        if (dsequence_history == nullptr) {
            // Free previously allocated memory (if any)
            cudaFree(dw_q);
            cudaFree(dw_k);
            cudaFree(dw_v);
            cudaFree(dmask);
            cudaFree(dsequence_history);
            cudaFree(dattention_scores);
            cudaFree(dQ);
            cudaFree(dV);
            cudaFree(dOut);
            cudaFree(dK);
            cudaFree(dhead_out);
            cudaFree(dattention_out);

            // Allocate device memory
            cudaMalloc((void**)&dsequence_history, m * k * sizeof(float));
            checkCudaError("cudaMalloc for dsequence_history");
            
            cudaMalloc((void**)&dw_q, k * d_model * sizeof(float));
            checkCudaError("cudaMalloc for dw_q");
            
            cudaMalloc((void**)&dw_k, k * d_model * sizeof(float));
            checkCudaError("cudaMalloc for dw_k");
            
            cudaMalloc((void**)&dw_v, k * d_model * sizeof(float));
            checkCudaError("cudaMalloc for dw_v");
            
            cudaMalloc((void**)&dQ, sequence_len * d_model * sizeof(float));
            checkCudaError("cudaMalloc for dQ");
            
            cudaMalloc((void**)&dV, sequence_len * d_model * sizeof(float));
            checkCudaError("cudaMalloc for dV");
            
            cudaMalloc((void**)&dK, sequence_len * d_model * sizeof(float));
            checkCudaError("cudaMalloc for dK");
            
            cudaMalloc((void**)&dOut, sequence_len * d_model * sizeof(float));
            checkCudaError("cudaMalloc for dOut");
            
            cudaMalloc((void**)&dattention_scores, num_ma_heads*m * m * sizeof(float));
            checkCudaError("cudaMalloc for dattention_scores");

            cudaMalloc((void**)&dattention_scores_softmax, m * m * sizeof(float));
            checkCudaError("cudaMalloc for dattention_scores");
            
            cudaMalloc((void**)&dmask, m * m * sizeof(float));
            checkCudaError("cudaMalloc for dmask");
            
            // Allocate memory for head_out (stores attention-weighted V output)
            cudaMalloc((void**)&dhead_out, sequence_len * d_head * sizeof(float));
            checkCudaError("cudaMalloc for dhead_out");

            // Allocate memory for final attention_out (concatenated head_out results)
            cudaMalloc((void**)&dattention_out, sequence_len * d_model * sizeof(float));
            checkCudaError("cudaMalloc for dattention_out");
        }

    // Copy host data to device memory
    cudaMemcpyAsync(dmask, hmask, m * m * sizeof(float), cudaMemcpyHostToDevice,0);
    cudaMemcpyAsync(dsequence_history, hsequence_history, m * k * sizeof(float), cudaMemcpyHostToDevice,0);
    cudaMemcpyAsync(dw_q, hw_q, k * d_model * sizeof(float), cudaMemcpyHostToDevice,0);
    cudaMemcpyAsync(dw_v, hw_v, k * d_model * sizeof(float), cudaMemcpyHostToDevice,0);
    cudaMemcpyAsync(dw_k, hw_k, k * d_model * sizeof(float), cudaMemcpyHostToDevice,0);

    // Initialize device output matrices
    cudaMemsetAsync(dQ, 0, sequence_len * d_model * sizeof(float),0);
    cudaMemsetAsync(dV, 0, sequence_len * d_model * sizeof(float),0);
    cudaMemsetAsync(dK, 0, sequence_len * d_model * sizeof(float),0);
    cudaMemsetAsync(dattention_scores, 0, num_ma_heads*m * m * sizeof(float),0);
    cudaMemsetAsync(dattention_scores_softmax, 0, m * m * sizeof(float),0);
    cudaMemsetAsync(dhead_out, 0, sequence_len * d_head * sizeof(float),0);
    cudaMemsetAsync(dattention_out, 0, sequence_len * d_model * sizeof(float), 0);

    // Kernel configuration
    dim3 blockDim1(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim1((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);        
    sgemm_blockheads<<<gridDim1, blockDim1, 0, 0>>>(m, n, k, dsequence_history, dw_q, dw_v, dw_k, dQ, dV, dK);
    cudaDeviceSynchronize();

    dim3 blockSize(8, 8, 1);  // Adjust based on available resources
    dim3 gridSize(
    (sequence_len + blockSize.x - 1) / blockSize.x,
    (sequence_len + blockSize.y - 1) / blockSize.y,
     num_heads);
    // Each head gets its own block in the z-dimension

    compute_attention_scores<<<gridSize, blockSize>>>(dQ, dK, dmask, dattention_scores, sequence_len, d_head, use_mask, num_ma_heads, d_model);
    print_attention_scores_per_head(dattention_scores, num_heads, sequence_len);
    cudaDeviceSynchronize();
    // dim3 blockDim1(16, 16);
    // dim3 gridDim1((d_model + blockDim.x - 1) / blockDim.x, (d_model + blockDim.y - 1) / blockDim.y); 
    // for(int head = 0; head < num_ma_heads; ++head)
    // {
    //     compute_attention_scores<<<gridDim1, blockDim1,0,streams[head]>>>(dQ, dK, dmask, dattention_scores, sequence_len, d_head, use_mask, head, d_model);
    //     softMax<<<gridDim1, blockDim1,0,streams[head]>>>(dattention_scores_softmax,dattention_scores,sequence_len,sequence_len);
    //     compute_head_out<<<gridDim1, blockDim1,0,streams[head]>>>(dhead_out, dattention_scores_softmax,dV, sequence_len,d_model,head, d_head);
    //     cudaDeviceSynchronize();
    //     merge_heads<<<gridDim1, blockDim1,0,streams[head]>>>(dhead_out, dattention_out, sequence_len,d_model, num_ma_heads, d_head,head);
    //     // print_device_matrix(dattention_scores, sequence_len, sequence_len, "dattention_out",streams[0]);
    //     // print_device_matrix(dhead_out, sequence_len, d_head, "dhead_out",streams[0]);
    // }
    // print_device_matrix(dattention_out, sequence_len, d_model, "dattention_out",streams[0]);

        // Cleanup
        delete[] hsequence_history;
        delete[] hout;
        delete[] hmask;
        delete[] hw_q;
        delete[] hw_v;
        delete[] hw_k;

        for (int i = 0; i < num_ma_heads; i++)
            cudaStreamDestroy(streams[i]);
        
        return attention_out;
    }
    
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
        FixedVector<float> bias) 
        {

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
            cudaMallocAsync((void**)&dA, m * k * sizeof(float), 0);
            checkCudaError("CudaMalloc for dA");
            cudaMallocAsync((void**)&dB, k * n * sizeof(float), 0);
            checkCudaError("CudaMalloc for dB");
            cudaMallocAsync((void**)&dOut, m * n * sizeof(float), 0);
            checkCudaError("CudaMalloc for dOut");
            cudaMallocAsync((void**)&dBias, n * sizeof(float), 0);
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

        linear_kernel<<<gridDim, blockDim, m*n*sizeof(float), 0>>>(m, n, k, dA, dB, dBias, dOut);
        checkCudaError("Linear Kernel Failure");

        //start = std::chrono::high_resolution_clock::now();
        // 4.) Copy the result back to host
        cudaStreamSynchronize(0); // Sync default stram before using dOut on another stream
        cudaMemcpyAsync(hOut, dOut, m * n * sizeof(float), cudaMemcpyDeviceToHost, 0);
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