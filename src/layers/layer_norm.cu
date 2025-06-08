// src/layers/layer_norm.cu
#include "layer_norm.cuh"
#include <cuda_runtime.h>

__global__ void layer_norm_kernel(float *input, float *output, float *gamma, float *beta, 
                                   int N, int D, float epsilon) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float mean = 0.0f;
        float variance = 0.0f;

        // Calculate mean
        for (int j = 0; j < D; j++) {
            mean += input[idx * D + j];
        }
        mean /= D;

        // Calculate variance
        for (int j = 0; j < D; j++) {
            float diff = input[idx * D + j] - mean;
            variance += diff * diff;
        }
        variance /= D;

        float stddev = sqrt(variance + epsilon);

        // Normalize and apply gamma and beta
        for (int j = 0; j < D; j++) {
            output[idx * D + j] = gamma[j] * (input[idx * D + j] - mean) / stddev + beta[j];
        }
    }
}

void LayerNorm::forward(float *input, float *output, float *gamma, float *beta, 
                        int N, int D, float epsilon) {
    float *d_input, *d_output, *d_gamma, *d_beta;
    size_t input_size = N * D * sizeof(float);
    size_t output_size = N * D * sizeof(float);
    size_t gamma_size = D * sizeof(float);
    size_t beta_size = D * sizeof(float);

    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, output_size);
    cudaMalloc(&d_gamma, gamma_size);
    cudaMalloc(&d_beta, beta_size);

    cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, gamma, gamma_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, beta, beta_size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    layer_norm_kernel<<<numBlocks, blockSize>>>(d_input, d_output, d_gamma, d_beta, N, D, epsilon);

    cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_gamma);
    cudaFree(d_beta);
}