// filepath: cuda-transformer/cuda-transformer/src/layers/feed_forward.cu
#include "feed_forward.cuh"
#include "utils/cuda_utils.cuh"

__global__ void feedForwardKernel(const Matrix input, Matrix output, const Matrix W1, const std::vector<double> b1, const Matrix W2, const std::vector<double> b2, size_t d_ff) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output.getRows()) {
        // First layer: Linear + ReLU
        for (int j = 0; j < d_ff; ++j) {
            double sum = 0.0;
            for (int k = 0; k < input.getCols(); ++k) {
                sum += input[idx][k] * W1[k][j];
            }
            sum += b1[j];
            output[idx][j] = max(0.0, sum); // ReLU activation
        }

        // Second layer: Linear
        for (int j = 0; j < output.getCols(); ++j) {
            double sum = 0.0;
            for (int k = 0; k < d_ff; ++k) {
                sum += output[idx][k] * W2[k][j];
            }
            sum += b2[j];
            output[idx][j] = sum; // No activation
        }
    }
}

Matrix FeedForward::forward(const Matrix &input) {
    Matrix output(input.getRows(), d_ff);
    Matrix W1 = this->W1; // Assuming W1 is already initialized
    Matrix W2 = this->W2; // Assuming W2 is already initialized

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (output.getRows() + blockSize - 1) / blockSize;
    feedForwardKernel<<<numBlocks, blockSize>>>(input, output, W1, b1, W2, b2, d_ff);
    cudaDeviceSynchronize();

    return output;
}