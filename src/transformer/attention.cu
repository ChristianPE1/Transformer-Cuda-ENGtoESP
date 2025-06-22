// filepath: /cuda-transformer/cuda-transformer/src/transformer/attention.cu
#include "attention.cuh"
#include "utils/cuda_utils.cuh"
#include <cmath>

__device__ void softmax(float* data, int length) {
    float max_val = data[0];
    for (int i = 1; i < length; ++i) {
        if (data[i] > max_val) {
            max_val = data[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < length; ++i) {
        data[i] = exp(data[i] - max_val);
        sum += data[i];
    }

    for (int i = 0; i < length; ++i) {
        data[i] /= sum;
    }
}

__global__ void multiHeadAttentionKernel(float* queries, float* keys, float* values, 
                                         float* output, int d_model, int n_heads, 
                                         int seq_length) {
    int head_size = d_model / n_heads;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < seq_length) {
        float attention_scores[seq_length];
        for (int i = 0; i < seq_length; ++i) {
            attention_scores[i] = 0.0f;
            for (int j = 0; j < head_size; ++j) {
                attention_scores[i] += queries[idx * d_model + j] * keys[i * d_model + j];
            }
        }

        softmax(attention_scores, seq_length);

        for (int i = 0; i < seq_length; ++i) {
            output[idx * d_model + i] = 0.0f;
            for (int j = 0; j < head_size; ++j) {
                output[idx * d_model + i] += attention_scores[j] * values[i * d_model + j];
            }
        }
    }
}

Matrix MultiHeadAttention::forward(const Matrix &query, const Matrix &key, const Matrix &value, const Matrix &mask) {
    int seq_length = query.getRows();
    int d_model = query.getCols();

    Matrix output(seq_length, d_model);

    // Llama al kernel usando los datos de los Matrix
    int blockSize = 256;
    int numBlocks = (seq_length + blockSize - 1) / blockSize;
    multiHeadAttentionKernel<<<numBlocks, blockSize>>>(
        query.getData(),
        key.getData(),
        value.getData(),
        output.getData(),
        d_model,
        n_heads,
        seq_length
    );
    cudaDeviceSynchronize();

    return output;
}