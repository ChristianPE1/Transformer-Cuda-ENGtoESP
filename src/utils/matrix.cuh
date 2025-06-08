#ifndef MATRIX_CUH
#define MATRIX_CUH

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

class Matrix
{
private:
    float *data;
    int rows, cols;

public:
    Matrix(int rows, int cols);
    Matrix(int rows, int cols, float init_val);
    ~Matrix();

    // Basic operations
    void initialize(float value = 0.0f);
    void copyFromHost(const float *hostData);
    void copyToHost(float *hostData) const;

    // Getters
    int getRows() const { return rows; }
    int getCols() const { return cols; }
    float *getData() const { return data; }

    // Matrix operations
    Matrix add(const Matrix &other) const;
    Matrix multiply(const Matrix &other) const;
    Matrix scale(float scalar) const;

    // Element access
    float *operator[](int row) const;
};

#endif // MATRIX_CUH