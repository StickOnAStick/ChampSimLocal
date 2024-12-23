#ifndef FIXED_VECTOR_MATH_H
#define FIXED_VECTOR_MATH_H

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

        FixedVector<FixedVector<T>> transposed(cols);
        for (std::size_t i = 0; i < cols; i++) {
            transposed[i] = FixedVector<T>(rows);
            for (std::size_t j = 0; j < rows; j++) {
                transposed[i][j] = matrix[j][i];
            }
        }
        return transposed;
    }

    // Normalize the fixed vector
    template <typename T>
    void normalize(FixedVector<T>& vec) {
        T sum_of_squares = 0;
        for (std::size_t i = 0; i < vec.size(); i++) {
            sum_of_squares += vec[i] * vec[i];
        }

        T magnitude = std::sqrt(sum_of_squares);
        
        if (magnitude == 0) {
            throw std::runtime_error("Cannot normalize a vector with zero magnitude.");
        }

        for (std::size_t i = 0; i < vec.size(); i++) {
            vec[i] /= magnitude;
        }
    }

    // Normalize each row in the matrix
    // In transformer logic, we do normalization by row
    template <typename T>
    void normalize(FixedVector<FixedVector<T>>& matrix) {
        for (std::size_t i = 0; i < matrix.size(); i++) {
            normalize(matrix[i]);
        }
    }

    // Dot product of matrix
    template <typename T>
    FixedVector<FixedVector<T>> dotProduct(const FixedVector<FixedVector<T>>& A, const FixedVector<FixedVector<T>>& B) {
        size_t rowsA = A.size();
        size_t colsA = A[0].size();
        size_t colsB = B[0].size();

        if (rowsA != colsB){
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

    // Softmax 1D
    template <typename T>
    void softmax(FixedVector<T>& matrix) {
        T maxVal = *std::max_element(matrix.begin(), matrix.end());
        T sumExp = 0;

        for (std::size_t i = 0; i < matrix.size(); i++) {
            matrix[i] = std::exp(matrix[i] - maxVal); // Stability adjustment
            sumExp += matrix[i];
        }

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
                scores[i][j] = -std::numeric_limits<T>::infinity();
            }
        }
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

        auto *b_it = B.begin();
        for (FixedVector<T>& a : A){
            FixedVectorMath::add(a, *b_it);
            ++b_it;
        }
    }

}

#endif // FIXED_VECTOR_MATH_H