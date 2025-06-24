#include "utils/matrix.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <stdexcept>

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
    // Implementación simplificada
    Matrix result(rows, other.cols);
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

// ===== OPERACIONES BATCH OPTIMIZADAS =====

void Matrix::copyToHostBatch(std::vector<float>& host_data) const {
    host_data.resize(rows * cols);
    cudaMemcpy(host_data.data(), data, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
}

void Matrix::copyFromHostBatch(const std::vector<float>& host_data) {
    if (host_data.size() != rows * cols) return;
    cudaMemcpy(data, host_data.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice);
}

// ===== KERNELS CUDA OPTIMIZADOS =====

__global__ void matrixMultiplyKernel(const float* A, const float* B, float* C, 
                                     int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

Matrix Matrix::matrixMultiply(const Matrix& other) const {
    if (cols != other.rows) {
        throw std::runtime_error("Matrix dimensions incompatible for multiplication");
    }
    
    Matrix result(rows, other.cols);
    
    dim3 blockSize(16, 16);
    dim3 gridSize((other.cols + blockSize.x - 1) / blockSize.x,
                  (rows + blockSize.y - 1) / blockSize.y);
    
    matrixMultiplyKernel<<<gridSize, blockSize>>>(

        data, other.data, result.data, rows, other.cols, cols);
    
    cudaDeviceSynchronize();
    return result;
}

__global__ void softmaxKernel(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        // Find max
        float max_val = input[row * cols];
        for (int j = 1; j < cols; ++j) {
            float val = input[row * cols + j];
            if (val > max_val) max_val = val;
        }
        
        // Compute exp and sum
        float sum = 0.0f;
        for (int j = 0; j < cols; ++j) {
            float exp_val = expf(input[row * cols + j] - max_val);
            output[row * cols + j] = exp_val;
            sum += exp_val;
        }
        
        // Normalize
        for (int j = 0; j < cols; ++j) {
            output[row * cols + j] /= sum;
        }
    }
}

Matrix Matrix::softmax() const {
    Matrix result(rows, cols);
    
    int blockSize = 256;
    int numBlocks = (rows + blockSize - 1) / blockSize;
    
    softmaxKernel<<<numBlocks, blockSize>>>(data, result.data, rows, cols);
    cudaDeviceSynchronize();
    
    return result;
}

__global__ void crossEntropyGradKernel(const float* predictions, const float* targets, 
                                       float* gradients, int batch_size, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size * num_classes) {
        int row = idx / num_classes;
        int col = idx % num_classes;
        int target_class = (int)targets[row];
        
        if (col == target_class) {
            gradients[idx] = predictions[idx] - 1.0f;
        } else {
            gradients[idx] = predictions[idx];
        }
    }
}

Matrix Matrix::crossEntropyGrad(const Matrix& targets) const {
    Matrix grad(rows, cols);
    
    int total_elements = rows * cols;
    int blockSize = 256;
    int numBlocks = (total_elements + blockSize - 1) / blockSize;
    
    crossEntropyGradKernel<<<numBlocks, blockSize>>>(

        data, targets.data, grad.data, rows, cols);
    
    cudaDeviceSynchronize();
    return grad;
}