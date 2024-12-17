#include <iostream>
#include "../inc/FixedVectorMath.hh"
#include "../inc/FixedVector.hh"

template <typename T>
void Print1DVec(const FixedVector<T>& v) {
    std::cout << "1D Vector of type: " << typeid(T).name() << std::endl;

    // Print each element of the vector
    for (size_t i = 0; i < v.size(); ++i) {
        std::cout << v[i] << " ";
    }

    std::cout << std::endl;
}

template <typename T>
void Print2DVec(const FixedVector<FixedVector<T>>& vec2D) {
    std::cout << "2D Vector of type: " << typeid(T).name() << std::endl;

    // Loop through each row of the 2D vector
    for (size_t i = 0; i < vec2D.size(); ++i) {
        const FixedVector<T>& row = vec2D[i]; // Get a row
        for (size_t j = 0; j < row.size(); ++j) {
            std::cout << row[j] << " "; // Print each element in the row
        }
        std::cout << std::endl; // Newline after each row
    }

    std::cout << "End of 2D Vector" << std::endl;
}

void Normalize() {
    std::cout << "-----Normalize Test-----" << std::endl;

    // Arrange
    size_t size = 4;
    FixedVector<float> floatVector(size, 0);
    for (int i = 0; i < floatVector.size(); i++) {
        floatVector[i] = 3;
    }
    Print1DVec(floatVector);

    // Act
    FixedVectorMath::normalize(floatVector);

    // Assert
    std::cout << "Normalize Test Output" << std::endl;
    Print1DVec(floatVector);
}

void Transpose() {
    std::cout << "-----Transpose Test-----" << std::endl;

    // Arrange
    size_t rowsA = 2;
    size_t colsB = 6;
    FixedVector<FixedVector<float>> floatVector(rowsA, FixedVector<float>(colsB, 0));
    float amt = 0.0f;
    for (int i = 0; i < floatVector.size(); i++) {
        for (int j = 0; j < floatVector[i].size(); j++) {
            floatVector[i][j] = amt;
            amt += 0.8f;
        }
    }
    Print2DVec(floatVector);

    // Act
    FixedVector<FixedVector<float>> outputVector = FixedVectorMath::transpose(floatVector);

    // Assert
    std::cout << "Transpose Test Output" << std::endl;
    Print2DVec(outputVector);
}

void ApplyMask() {
    std::cout << "-----Apply Mask Test-----" << std::endl;

    // Arrange
    size_t rowsA = 2;
    size_t colsB = 6;
    FixedVector<FixedVector<float>> floatVector(rowsA, FixedVector<float>(colsB, 0));
    float amt = 0.0f;
    for (int i = 0; i < floatVector.size(); i++) {
        for (int j = 0; j < floatVector[i].size(); j++) {
            floatVector[i][j] = amt;
            amt += 0.8f;
        }
    }
    Print2DVec(floatVector);

    FixedVector<FixedVector<float>> maskVector(rowsA, FixedVector<float>(colsB, 0));
    for (int i = 0; i < maskVector.size(); i++) {
        for (int j = 0; j < maskVector[i].size(); j++) {
            maskVector[i][j] = ((i + j) % 2 == 0) ? 1.0f : 0.0f;
        }
    }
    Print2DVec(maskVector);

    // Act
    FixedVectorMath::applyMask(floatVector, maskVector);

    // Assert
    std::cout << "Apply Mask Test Output" << std::endl;
    Print2DVec(floatVector);
}

void SoftMax() {
    std::cout << "-----Softmax Test-----" << std::endl;

    // Arrange
    size_t rowsA = 2;
    size_t colsB = 6;
    FixedVector<FixedVector<float>> floatVector(rowsA, FixedVector<float>(colsB, 0));
    float amt = 0.0f;
    for (int i = 0; i < floatVector.size(); i++) {
        for (int j = 0; j < floatVector[i].size(); j++) {
            floatVector[i][j] = amt;
            amt += 0.8f;
        }
    }
    Print2DVec(floatVector);

    // Act
    FixedVectorMath::softmax(floatVector);

    // Assert
    std::cout << "Softmax Test Output" << std::endl;
    Print2DVec(floatVector);
}

void DotProduct() {
    std::cout << "-----Dot Product Test-----" << std::endl;

    // Arrange
    size_t rowsA = 2;
    size_t colsB = 6;
    FixedVector<FixedVector<float>> AVector(rowsA, FixedVector<float>(colsB, 0));
    float amt = 0.0f;
    for (int i = 0; i < AVector.size(); i++) {
        for (int j = 0; j < AVector[i].size(); j++) {
            AVector[i][j] = amt;
            amt += 0.8f;
        }
    }
    Print2DVec(AVector);

    FixedVector<FixedVector<float>> BVector(rowsA, FixedVector<float>(colsB, 0));
    float amt2 = 9.0f;
    for (int i = 0; i < BVector.size(); i++) {
        for (int j = 0; j < BVector[i].size(); j++) {
            BVector[i][j] = amt2;
            amt2 -= 1.56f;
        }
    }
    Print2DVec(BVector);

    // Act
    FixedVector<FixedVector<float>> OutVector = FixedVectorMath::dotProduct(AVector, BVector);

    // Assert
    std::cout << "Dot Product Test Output" << std::endl;
    Print2DVec(OutVector);
}

int main() {
    // Coverage Tests
    Normalize();
    Transpose();
    ApplyMask();
    SoftMax();
    DotProduct();

    std::cout << "Unit Testing Completed\n";
    return 0;
}
