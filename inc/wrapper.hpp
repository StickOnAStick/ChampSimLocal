// wrapper.hpp
#ifndef WRAPPER_HPP
#define WRAPPER_HPP
#include "FixedVector.hh"

namespace CudaFixedVectorMath {

void mul(FixedVector<float>& out,
         FixedVector<float>& A, 
         FixedVector<float>& B); 

void mul(FixedVector<FixedVector<float>>& out,
         FixedVector<FixedVector<float>>& A, 
         FixedVector<FixedVector<float>>& B); 

void add(FixedVector<float>& A,
         FixedVector<float>& B);

void add(FixedVector<FixedVector<float>>& A, 
         FixedVector<FixedVector<float>>& B);

FixedVector<FixedVector<float>> dotproduct(
            FixedVector<FixedVector<float>>& A,
            FixedVector<FixedVector<float>>& B);

FixedVector<FixedVector<float>> linear(
            FixedVector<FixedVector<float>> A, 
            FixedVector<FixedVector<float>> B, 
            FixedVector<float> bias);
}

#endif // WRAPPER_HPP