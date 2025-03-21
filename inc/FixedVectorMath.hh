#ifndef FIXED_VECTOR_MATH_H
#define FIXED_VECTOR_MATH_H
#pragma once

#include "FixedVector.hh"
#include <cmath> // For sqrt and pow
#include <algorithm>
#include <stdexcept>

namespace FixedVectorMath {
    // Transpose operation for a matrix
    template <typename T>
    FixedVector<FixedVector<T>> transpose(const FixedVector<FixedVector<T>>& matrix) {
        std::size_t rows = matrix.size();
        std::size_t cols = matrix[0].size();

        // Create the transposed matrix with 'cols' rows
        FixedVector<FixedVector<T>> transposed(cols);

        // Initialize each row of the transposed matrix
        for (std::size_t i = 0; i < cols; i++)
        {
            // Explicitly create a FixedVector<T> for each row of the transposed matrix
            transposed[i] = FixedVector<T>(rows);  // No ambiguity here, as we are specifying the constructor

            // Transpose the elements
            for (std::size_t j = 0; j < rows; j++)
            {
                transposed[i][j] = matrix[j][i];
            }
        }

        return transposed;
    }

    template <typename T>
    void normalize(
        FixedVector<FixedVector<T>>& matrix,
        FixedVector<T>& means,
        FixedVector<T>& vars,
        T epsilon = 1e-5
    ) {
        // We do LN row-by-row. Each row is [d_model] wide.
        // The means[] and vars[] must be sized == matrix.size().
    
        if (matrix.size() != means.size() || matrix.size() != vars.size()) {
            throw std::runtime_error("Mismatch in dimension for LN forward: matrix vs means/vars.");
        }
    
        // For each row i in [0..sequence_len-1]
        for (size_t i = 0; i < matrix.size(); i++) {
            // 1) Compute mean over row i
            T sumVal = 0.0f;
            for (size_t k = 0; k < matrix[i].size(); k++) {
                sumVal += matrix[i][k];
            }
            T mean = sumVal / static_cast<T>(matrix[i].size());
            means[i] = mean; // store for backprop
    
            // 2) Compute variance
            T varVal = 0.0f;
            for (size_t k = 0; k < matrix[i].size(); k++) {
                T diff = matrix[i][k] - mean;
                varVal += diff * diff;
            }
            varVal /= static_cast<T>(matrix[i].size());
            vars[i] = varVal; // store for backprop
    
            // 3) Normalize each element
            T invStd = 1.0f / std::sqrt(varVal + epsilon);
            for (size_t k = 0; k < matrix[i].size(); k++) {
                matrix[i][k] = (matrix[i][k] - mean) * invStd;
            }
        }
    }

    // Dot product of matrix
    template <typename T>
    FixedVector<FixedVector<T>> dot(const FixedVector<FixedVector<T>>& A, const FixedVector<FixedVector<T>>& B) {
        size_t rowsA = A.size();
        size_t rowsB = B.size();
        size_t colsA = A[0].size();
        size_t colsB = B[0].size();

        if (colsA != rowsB){ // [a, n] * [n, b] is only valid solution
            throw std::invalid_argument("Rows of matrix A does not match Cols of matrix B");
        }

        FixedVector<FixedVector<T>> result(rowsA, FixedVector<T>(colsB, static_cast<T>(0)));

        for (std::size_t i = 0; i < rowsA; ++i) {
            for (std::size_t j = 0; j < colsB; ++j) {
                for (std::size_t k = 0; k < colsA; ++k) {
                    result[i][j] += A[i][k] * B[k][j];
                }
            }
        }

        return result;
    }

    template <typename T>
    FixedVector<FixedVector<T>> linear(
        const FixedVector<FixedVector<T>>& A, 
        const FixedVector<FixedVector<T>>& B, 
        const FixedVector<T>& bias
    )
    {
        // Rows != Cols
        if (A[0].size() != B.size()) throw std::invalid_argument("Cannot perform 2D linear on matricies of different sizes");
        
        FixedVector<FixedVector<T>> res(A.size(), FixedVector<T>(B[0].size(), static_cast<T>(0)));
        for(size_t rows = 0; rows < A.size(); ++rows){
            for(size_t cols = 0; cols < B[0].size(); ++cols){
                for(size_t m = 0; m < A[0].size(); ++m){
                    res[rows][cols]+= A[rows][m] * B[m][cols];
                }
                res[rows][cols] += bias[cols];
            }
        }
        return res;
    }

    // Softmax 1D
    template <typename T>
    void softmax(FixedVector<T>& matrix) {
        T maxVal = *std::max_element(matrix.begin(), matrix.end());
        T sumExp = 0;

        for (std::size_t i = 0; i < matrix.size(); i++) {
            matrix[i] = std::exp(matrix[i] - maxVal); // Stability adjustment
            sumExp += matrix[i];
        }
        sumExp = std::max(sumExp, std::numeric_limits<T>::epsilon());
        for (std::size_t i = 0; i < matrix.size(); i++) {
            matrix[i] /= sumExp;
        }
    }

    template<typename T>
    void mul(FixedVector<T>& out, const FixedVector<T>& A, const FixedVector<T>& B){
        if (A.size() != B.size() || A.size() != out.size())
            throw std::invalid_argument("Size mismatch between out = A*B matricies.");
        
        for(std::size_t i = 0; i < out.size(); ++i){
            out[i] = A[i] * B[i]; 
        }
    }

    template<typename T>
    void mul(
        FixedVector<FixedVector<T>>& out, 
        const FixedVector<FixedVector<T>>& A, 
        const FixedVector<FixedVector<T>>& B
    ){
        if (A.size() != B.size() || A.size() != out.size())
            throw std::invalid_argument("Size mismatch between out = A*B matricies.");
        
        for(size_t i = 0; i  < A.size(); ++i){
            mul(out[i], A[i], B[i]);
        }
    }

    // Softmax
    template <typename T>
    void softmax(FixedVector<FixedVector<T>>& matrix) {
        for (std::size_t i = 0; i < matrix.size(); ++i) {
            softmax(matrix[i]);
        }
    }

    // Apply mask, used in Masked Multi Headed Attention
    template <typename T>
    void applyMask(FixedVector<FixedVector<T>>& scores) {
        // mask = [ [0, -inf, -inf, -inf, -inf],
        //          [0,    0, -inf, -inf, -inf],
        //          [0,    0,    0, -inf, -inf],
        //          [0,    0,    0,    0, -inf],
        //          [0,    0,    0,    0,    0] ]
        for (std::size_t i = 0; i < scores.size(); ++i) {
            for (std::size_t j = i + 1; j < scores[i].size(); ++j) {
                scores[i][j] = std::numeric_limits<T>::lowest();
            }
        }
    }

    // A simple CPU-based matrix multiply: C = A x B
    // A is [M x K], B is [K x N], result is [M x N].
    template <typename T>
    FixedVector<FixedVector<T>> matrixMultiplyCPU(
        const FixedVector<FixedVector<T>>& A,
        const FixedVector<FixedVector<T>>& B
    ) {
        // 1) Validate shapes
        int M = (int)A.size();
        if (M == 0) {
            return FixedVector<FixedVector<float>>{};
        }
        int K = (int)A[0].size();  
        for (int i = 1; i < M; i++) {
            if ((int)A[i].size() != K) {
                throw std::runtime_error("matrixMultiplyCPU: Inconsistent row size in A.");
            }
        }

        if ((int)B.size() != K) {
            throw std::runtime_error("matrixMultiplyCPU: B must have K rows.");
        }
        int N = (int)B[0].size();
        for (int i = 1; i < K; i++) {
            if ((int)B[i].size() != N) {
                throw std::runtime_error("matrixMultiplyCPU: Inconsistent row size in B.");
            }
        }

        // 2) Allocate result [M x N], initialized to 0
        FixedVector<FixedVector<float>> C(M, FixedVector<float>(N, 0.0f));

        // 3) Triple nested loop: for each row i, col j, sum over k
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += A[i][k] * B[k][j];
                }
                C[i][j] = sum;
            }
        }

        return C;
    }

    template <typename T>
    void add(FixedVector<T>& A, FixedVector<T>& B){
        /*
            This variant stores the result in A
        */
        if(A.size() != B.size()){
            throw std::invalid_argument("Matricies cannot be of different sizes!");
        }

        auto b_it = B.begin(); // Don't hate me, I'm praciticng __iterators__ in cpp
        for(T& a : A){
            a += *b_it;
            ++b_it;
        }
    }

    template <typename T>
    void add(FixedVector<FixedVector<T>>& A, FixedVector<FixedVector<T>>& B){
        /*
        NOTE: This variant stores the result in A
        */
        if(A.size() != B.size())
            throw std::invalid_argument("Matricies cannot be of different sizes!");

        auto b_it = B.begin();
        for (FixedVector<T>& a : A){
            FixedVectorMath::add(a, *b_it);
            ++b_it;
        }
    }

    inline float relu(float in){
        if (in < 0)
            return 0;
        
        return in;
    }
    // Not templating :>
    inline void relu(FixedVector<FixedVector<float>>& A){
        /*
            In-place relu of 2D mat
        */
       for(size_t i = 0; i < A.size(); ++i){
        for(size_t j = 0; j < A[0].size(); ++j){
            A[i][j] = FixedVectorMath::relu(A[i][j]);
        }
       }
    }

    
    // Forward declarations for CUDA API method defintions
    // Prevents linker errors when building.
    FixedVector<FixedVector<float>> dotCuda(
        const FixedVector<FixedVector<float>>& A,
        const FixedVector<FixedVector<float>>& B
    );

    FixedVector<FixedVector<float>> linearCuda(
        FixedVector<FixedVector<float>> A,
        FixedVector<FixedVector<float>> B,
        FixedVector<float> bias
    );

    void addCuda(
        FixedVector<float>& A,
        FixedVector<float>& B
    );

    void addCuda(
        FixedVector<FixedVector<float>>& A,
        FixedVector<FixedVector<float>>& B
    );

    void normalizeCuda(
        FixedVector<FixedVector<float>>& matrix,
        FixedVector<float>& means,
        FixedVector<float>& vars,
        float epsilon = 1e-5f
    );
}

#endif // FIXED_VECTOR_MATH_H