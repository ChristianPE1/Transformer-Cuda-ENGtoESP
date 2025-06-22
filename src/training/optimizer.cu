// src/training/optimizer.cu
#include "optimizer.cuh"
#include "utils/cuda_utils.cuh"

__global__ void updateWeightsKernel(float *weights, float *gradients, float learning_rate, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= learning_rate * gradients[idx];
    }
}

Optimizer::Optimizer(float learning_rate) : learning_rate(learning_rate) {}

void Optimizer::update(float *weights, float *gradients, int size) {
    float *d_weights, *d_gradients;

    cudaMalloc(&d_weights, size * sizeof(float));
    cudaMalloc(&d_gradients, size * sizeof(float));

    cudaMemcpy(d_weights, weights, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gradients, gradients, size * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    updateWeightsKernel<<<numBlocks, blockSize>>>(d_weights, d_gradients, learning_rate, size);

    cudaMemcpy(weights, d_weights, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_weights);
    cudaFree(d_gradients);
}