#include <iostream>
#include "../inc/FixedVectorMath.hh"

template <typename T>
void Print1DVec(const FixedVector<T>& v) {
    std::cout << "1D Vector of type: " << typeid(T).name() << std::endl;

    // Print each element of the vector
    for (size_t i = 0; i < v.size(); ++i) {
        std::cout << v[i] << " ";
    }

    std::cout << std::endl;
}

int main() {
    std::cout << "TEST 1: Normalize" << std::endl;

    size_t size = 4;
    FixedVector<float> intVector(size, 0);
    Print1DVec(intVector);
    FixedVectorMath::normalize(intVector);
    Print1DVec(intVector);

    std::cout << "Unit Testing Completed\n";
    return 0;
}
