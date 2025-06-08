#include "matrix.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

__global__ void matrixAddKernel(float *a, float *b, float *result, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        result[idx] = a[idx] + b[idx];
    }
}

__global__ void matrixMultiplyKernel(float *a, float *b, float *result,
                                     int rows_a, int cols_a, int cols_b)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows_a && col < cols_b)
    {
        float sum = 0.0f;
        for (int k = 0; k < cols_a; k++)
        {
            sum += a[row * cols_a + k] * b[k * cols_b + col];
        }
        result[row * cols_b + col] = sum;
    }
}

Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols), on_device(true)
{
    cudaMalloc(&data, rows * cols * sizeof(float));
    cudaMemset(data, 0, rows * cols * sizeof(float));
}

Matrix::Matrix(int rows, int cols, float init_val) : rows(rows), cols(cols), on_device(true)
{
    cudaMalloc(&data, rows * cols * sizeof(float));

    // Initialize with value
    std::vector<float> host_data(rows * cols, init_val);
    cudaMemcpy(data, host_data.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice);
}

Matrix::~Matrix()
{
    if (data && on_device)
    {
        cudaFree(data);
    }
}

Matrix::Matrix(const Matrix &other) : rows(other.rows), cols(other.cols), on_device(true)
{
    cudaMalloc(&data, rows * cols * sizeof(float));
    cudaMemcpy(data, other.data, rows * cols * sizeof(float), cudaMemcpyDeviceToDevice);
}

Matrix &Matrix::operator=(const Matrix &other)
{
    if (this != &other)
    {
        if (data && on_device)
        {
            cudaFree(data);
        }

        rows = other.rows;
        cols = other.cols;
        on_device = true;

        cudaMalloc(&data, rows * cols * sizeof(float));
        cudaMemcpy(data, other.data, rows * cols * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    return *this;
}

Matrix Matrix::add(const Matrix &other) const
{
    if (rows != other.rows || cols != other.cols)
    {
        throw std::runtime_error("Matrix dimensions don't match for addition");
    }

    Matrix result(rows, cols);
    int size = rows * cols;

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    matrixAddKernel<<<numBlocks, blockSize>>>(data, other.data, result.data, size);
    cudaDeviceSynchronize();

    return result;
}

Matrix Matrix::multiply(const Matrix &other) const
{
    if (cols != other.rows)
    {
        throw std::runtime_error("Matrix dimensions don't match for multiplication");
    }

    Matrix result(rows, other.cols);

    dim3 blockSize(16, 16);
    dim3 gridSize((other.cols + blockSize.x - 1) / blockSize.x,
                  (rows + blockSize.y - 1) / blockSize.y);

    matrixMultiplyKernel<<<gridSize, blockSize>>>(
        data, other.data, result.data, rows, cols, other.cols);
    cudaDeviceSynchronize();

    return result;
}

void Matrix::copyFromHost(const std::vector<float> &hostData)
{
    if (hostData.size() != rows * cols)
    {
        throw std::runtime_error("Host data size doesn't match matrix size");
    }

    cudaMemcpy(data, hostData.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice);
}

void Matrix::copyToHost(std::vector<float> &hostData) const
{
    hostData.resize(rows * cols);
    cudaMemcpy(hostData.data(), data, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
}