#ifndef FIXED_VECTOR_MATH_H
#define FIXED_VECTOR_MATH_H

#include "../utils/FixedVector.hh"
#include "FixedVectorMath.hh"
#include <cmath>

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
    
}

#endif // FIXED_VECTOR_MATH_H