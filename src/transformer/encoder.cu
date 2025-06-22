// filepath: cuda-transformer/cuda-transformer/src/transformer/encoder.cu
#include "encoder.cuh"
#include "attention.cuh"
#include "layers/feed_forward.cuh"
#include "layers/layer_norm.cuh"
#include "utils/cuda_utils.cuh"

// Kernel function to launch multiple encoder layers
__global__ void encodeKernel(Matrix *input, Matrix *output, Matrix *src_mask, EncoderLayer *layers, size_t n_layers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_layers) {
        output[idx] = layers[idx].forward(input[idx], &src_mask[idx]);
    }
}

// Constructor de Encoder
Encoder::Encoder(size_t d_model, size_t n_heads, size_t n_layers, size_t d_ff)
    : n_layers(n_layers) {
    layers.reserve(n_layers);
    for (size_t i = 0; i < n_layers; ++i) {
        layers.emplace_back(d_model, n_heads, d_ff);
    }
}

void Encoder::forward(const Matrix &input, const Matrix &src_mask, Matrix &output) {
    // Launch kernel for encoding
    int blockSize = 256;
    int numBlocks = (n_layers + blockSize - 1) / blockSize;
    encodeKernel<<<numBlocks, blockSize>>>(input.device_ptr(), output.device_ptr(), src_mask.device_ptr(), layers.device_ptr(), n_layers);
    cudaDeviceSynchronize();
}