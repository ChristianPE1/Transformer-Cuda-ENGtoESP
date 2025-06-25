#include "utils/matrix.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <cstdlib>

__global__ void matrixAddKernel(float *a, float *b, float *result, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        result[idx] = a[idx] + b[idx];
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
    if (cols != other.rows) {
        throw std::runtime_error("Matrix dimensions don't match for multiplication");
    }

    Matrix result(rows, other.cols);
    
    // Simple CPU implementation for matrix multiplication
    std::vector<float> host_a(rows * cols);
    std::vector<float> host_b(other.rows * other.cols);
    copyToHost(host_a);
    other.copyToHost(host_b);
    
    std::vector<float> host_result(rows * other.cols, 0.0f);
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < other.cols; ++j) {
            for (int k = 0; k < cols; ++k) {
                host_result[i * other.cols + j] += host_a[i * cols + k] * host_b[k * other.cols + j];
            }
        }
    }
    
    result.copyFromHost(host_result);
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

float Matrix::getElement(int row, int col) const
{
    float value;
    cudaMemcpy(&value, &data[row * cols + col], sizeof(float), cudaMemcpyDeviceToHost);
    return value;
}

void Matrix::setElement(int row, int col, float value)
{
    cudaMemcpy(&data[row * cols + col], &value, sizeof(float), cudaMemcpyHostToDevice);
}

void Matrix::randomInitialize(float min_val, float max_val) {
    std::vector<float> host_data(rows * cols);
    for (int i = 0; i < rows * cols; ++i) {
        float random_val = ((float)rand() / RAND_MAX) * (max_val - min_val) + min_val;
        host_data[i] = random_val;
    }
    copyFromHost(host_data);
}

Matrix Matrix::transpose() const {
    Matrix result(cols, rows);
    
    // Simple CPU implementation for now
    std::vector<float> host_data(rows * cols);
    copyToHost(host_data);
    
    std::vector<float> transposed_data(cols * rows);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            transposed_data[j * rows + i] = host_data[i * cols + j];
        }
    }
    
    result.copyFromHost(transposed_data);
    return result;
}