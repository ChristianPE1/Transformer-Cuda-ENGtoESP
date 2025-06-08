// src/utils/matrix.cu
#include "matrix.cuh"
#include <cuda_runtime.h>
#include <iostream>

__global__ void matrixAddKernel(const float* A, const float* B, float* C, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        C[row * cols + col] = A[row * cols + col] + B[row * cols + col];
    }
}

Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols) {
    cudaMalloc(&data, rows * cols * sizeof(float));
}

Matrix::~Matrix() {
    cudaFree(data);
}

void Matrix::add(const Matrix& other, Matrix& result) {
    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);
    matrixAddKernel<<<gridSize, blockSize>>>(data, other.data, result.data, rows, cols);
    cudaDeviceSynchronize();
}

void Matrix::copyFromHost(const float* hostData) {
    cudaMemcpy(data, hostData, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
}

void Matrix::copyToHost(float* hostData) const {
    cudaMemcpy(hostData, data, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
}

void Matrix::initialize(float value) {
    float* hostData = new float[rows * cols];
    for (int i = 0; i < rows * cols; ++i) {
        hostData[i] = value;
    }
    copyFromHost(hostData);
    delete[] hostData;
}

int Matrix::getRows() const {
    return rows;
}

int Matrix::getCols() const {
    return cols;
}