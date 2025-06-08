// filepath: /cuda-transformer/cuda-transformer/src/transformer/embeddings.cu
#include "embeddings.cuh"
#include <cuda_runtime.h>

__global__ void embedKernel(float* embeddings, int* input_ids, float* output, int vocab_size, int d_model, int seq_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < seq_len) {
        for (int i = 0; i < d_model; ++i) {
            output[idx * d_model + i] = embeddings[input_ids[idx] * d_model + i];
        }
    }
}

Embedding::Embedding(int vocab_size, int d_model) : vocab_size(vocab_size), d_model(d_model) {
    cudaMalloc(&d_embeddings, vocab_size * d_model * sizeof(float));
    // Initialize embeddings here (e.g., with random values or pre-trained values)
}

Embedding::~Embedding() {
    cudaFree(d_embeddings);
}

void Embedding::forward(int* input_ids, float* output, int seq_len) {
    float* d_output;
    cudaMalloc(&d_output, seq_len * d_model * sizeof(float));

    int blockSize = 256;
    int numBlocks = (seq_len + blockSize - 1) / blockSize;
    embedKernel<<<numBlocks, blockSize>>>(d_embeddings, input_ids, d_output, vocab_size, d_model, seq_len);

    cudaMemcpy(output, d_output, seq_len * d_model * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_output);
}