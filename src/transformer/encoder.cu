// filepath: cuda-transformer/cuda-transformer/src/transformer/encoder.cu
#include "encoder.cuh"
#include "attention.cuh"
#include "feed_forward.cuh"
#include "layer_norm.cuh"
#include "cuda_utils.cuh"

__device__ Matrix EncoderLayer::forward(const Matrix &input, const Matrix &src_mask) {
    // Self-attention
    Matrix self_att_output = self_attention.forward(input, input, input, src_mask);
    Matrix norm1_output = norm1.forward(input.add(self_att_output));

    // Feed-forward
    Matrix ff_output = feed_forward.forward(norm1_output);
    Matrix norm2_output = norm2.forward(norm1_output.add(ff_output));

    return norm2_output;
}

EncoderLayer::EncoderLayer(size_t d_model, size_t n_heads, size_t d_ff)
    : self_attention(d_model, n_heads), feed_forward(d_model, d_ff),
      norm1(d_model), norm2(d_model) {}

// Kernel function to launch multiple encoder layers
__global__ void encodeKernel(Matrix *input, Matrix *output, Matrix *src_mask, EncoderLayer *layers, size_t n_layers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_layers) {
        output[idx] = layers[idx].forward(input[idx], src_mask[idx]);
    }
}

void Encoder::forward(const Matrix &input, const Matrix &src_mask, Matrix &output) {
    // Launch kernel for encoding
    int blockSize = 256;
    int numBlocks = (n_layers + blockSize - 1) / blockSize;
    encodeKernel<<<numBlocks, blockSize>>>(input.device_ptr(), output.device_ptr(), src_mask.device_ptr(), layers.device_ptr(), n_layers);
    cudaDeviceSynchronize();
}