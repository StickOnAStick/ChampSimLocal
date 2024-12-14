#ifndef FIXED_VECTOR_MATH_H
#define FIXED_VECTOR_MATH_H

#include "FixedVector.hh"
#include <cmath> // For sqrt and pow
#include <algorithm>

namespace FixedVectorMath {
    // Transpose operation for a matrix (2D FixedVector)
    template <typename T>
    FixedVector<FixedVector<T>> transpose(const FixedVector<FixedVector<T>>& matrix);

    // Normalize the fixed vector (1D FixedVector)
    template <typename T>
    void normalize(FixedVector<T>& vec);

    // Normalize each row in the matrix (2D FixedVector)
    template <typename T>
    void normalize(FixedVector<FixedVector<T>>& matrix);

}

#endif // FIXED_VECTOR_MATH_H
