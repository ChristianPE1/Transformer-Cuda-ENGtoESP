// filepath: cuda-transformer/cuda-transformer/src/transformer/embeddings.cuh
#ifndef EMBEDDINGS_H
#define EMBEDDINGS_H

#include <cuda_runtime.h>
#include "common.cuh"

class Embedding {
private:
    size_t vocab_size;
    size_t d_model;
    float* weights; // Pointer to weights on device

public:
    Embedding(size_t vocab_size, size_t d_model);
    ~Embedding();

    void initializeWeights();
    __device__ float* forward(const int* input_tokens, size_t seq_len);
};

#endif // EMBEDDINGS_H