#ifndef MATRIX_CUH
#define MATRIX_CUH

#include <cuda_runtime.h>
#include <vector>

class Matrix
{
private:
    float *data;
    int rows, cols;
    bool on_device;

public:
    __host__ __device__ Matrix(int rows, int cols);
    __host__ __device__ Matrix(int rows, int cols, float init_val);
    __host__ __device__ ~Matrix();

    // Copy constructor and assignment
    __host__ __device__ Matrix(const Matrix &other);
    __host__ __device__ Matrix &operator=(const Matrix &other);

    // Basic operations
    __host__ void initialize(float value = 0.0f);
    __host__ void copyFromHost(const std::vector<float> &hostData);
    __host__ void copyToHost(std::vector<float> &hostData) const;

    // Getters
    __host__ __device__ int getRows() const { return rows; }
    __host__ __device__ int getCols() const { return cols; }
    __host__ __device__ float *getData() const { return data; }

    // Matrix operations
    __host__ __device__ Matrix add(const Matrix &other) const;
    __host__ __device__ Matrix multiply(const Matrix &other) const;
    __host__ __device__ Matrix scale(float scalar) const;

    // Element access
    __host__ __device__ float &operator()(int row, int col);
    __host__ __device__ const float &operator()(int row, int col) const;
};

#endif // MATRIX_CUH