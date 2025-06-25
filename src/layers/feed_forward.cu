// filepath: cuda-transformer/cuda-transformer/src/layers/feed_forward.cu
#include "feed_forward.cuh"
#include "utils/cuda_utils.cuh"
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <algorithm>
#include <cmath>
#include <cstdlib> // Para rand()

__global__ void feedForwardKernel(
    const float* input, float* output,
    const float* W1, const float* b1,
    const float* W2, const float* b2,
    int rows, int input_dim, int d_ff, int output_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows) {
        // First layer: Linear + ReLU
        for (int j = 0; j < d_ff; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < input_dim; ++k) {
                sum += input[idx * input_dim + k] * W1[k * d_ff + j];
            }
            sum += b1[j];
            output[idx * d_ff + j] = fmaxf(0.0f, sum); // ReLU
        }

        // Second layer: Linear
        for (int j = 0; j < output_dim; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < d_ff; ++k) {
                sum += output[idx * d_ff + k] * W2[k * output_dim + j];
            }
            sum += b2[j];
            output[idx * output_dim + j] = sum; // No activation
        }
    }
}

Matrix FeedForward::forward(const Matrix &input) {
    int rows = input.getRows();
    int input_dim = input.getCols();
    
    // Create intermediate and output matrices
    Matrix intermediate(rows, d_ff);  // After first linear + ReLU
    Matrix output(rows, d_model);     // Final output
    
    // First layer: input -> d_ff with ReLU
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < d_ff; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < input_dim; ++k) {
                sum += input.getElement(i, k) * W1.getElement(k, j);
            }
            // Add bias and apply ReLU
            sum += (j < d_ff) ? 0.1f : 0.0f; // Simple bias
            intermediate.setElement(i, j, std::max(0.0f, sum)); // ReLU activation
        }
    }
    
    // Second layer: d_ff -> d_model (no activation)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < d_model; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < d_ff; ++k) {
                sum += intermediate.getElement(i, k) * W2.getElement(k, j);
            }
            // Add bias
            sum += (j < d_model) ? 0.05f : 0.0f; // Simple bias
            output.setElement(i, j, sum);
        }
    }

    return output;
}

FeedForward::FeedForward(size_t d_model, size_t d_ff) 
    : d_model(d_model), d_ff(d_ff), W1(d_model, d_ff), W2(d_ff, d_model) {
    initializeWeights();
    std::cout << "[FFN] Initialized with d_model=" << d_model << ", d_ff=" << d_ff << std::endl;
}

FeedForward::~FeedForward() {
    // Destructor cleanup if needed
}

void FeedForward::initializeWeights() {
    // Initialize W1 and W2 with Xavier/Glorot initialization
    float scale1 = sqrtf(2.0f / (d_model + d_ff));
    float scale2 = sqrtf(2.0f / (d_ff + d_model));
    
    W1.randomInitialize(-scale1, scale1);
    W2.randomInitialize(-scale2, scale2);
}

void FeedForward::updateWeights(float learning_rate) {
    // Simple weight updates with small random perturbations
    // In a full implementation, you'd use proper gradients
    float update_scale = learning_rate * 0.01f;
    
    for (int i = 0; i < W1.getRows(); ++i) {
        for (int j = 0; j < W1.getCols(); ++j) {
            float current = W1.getElement(i, j);
            float update = ((float)rand() / RAND_MAX - 0.5f) * update_scale;
            W1.setElement(i, j, current + update);
        }
    }
    
    for (int i = 0; i < W2.getRows(); ++i) {
        for (int j = 0; j < W2.getCols(); ++j) {
            float current = W2.getElement(i, j);
            float update = ((float)rand() / RAND_MAX - 0.5f) * update_scale;
            W2.setElement(i, j, current + update);
        }
    }
}