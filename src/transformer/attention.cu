// filepath: /cuda-transformer/cuda-transformer/src/transformer/attention.cu
#include "attention.cuh"
#include "utils/cuda_utils.cuh"
#include <cmath>

#define MAX_SEQ_LEN 256

__device__ void softmax(float* data, int length) {
    float max_val = data[0];
    for (int i = 1; i < length; ++i) {
        if (data[i] > max_val) max_val = data[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < length; ++i) {
        data[i] = expf(data[i] - max_val);
        sum += data[i];
    }
    for (int i = 0; i < length; ++i) {
        data[i] /= sum;
    }
}

__global__ void multiHeadAttentionKernel(
    const float* queries, const float* keys, const float* values, 
    float* output, int d_model, int n_heads, int seq_length, int key_seq_length) 
{
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int head_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int head_size = d_model / n_heads;

    if (token_idx < seq_length && head_idx < n_heads && key_seq_length <= MAX_SEQ_LEN) {
        float attention_scores[MAX_SEQ_LEN];
        
        // Calculate attention scores for this token and head
        for (int k = 0; k < key_seq_length; ++k) {
            float score = 0.0f;
            for (int d = 0; d < head_size; ++d) {
                int q_idx = token_idx * d_model + head_idx * head_size + d;
                int k_idx = k * d_model + head_idx * head_size + d;
                score += queries[q_idx] * keys[k_idx];
            }
            // Scale by sqrt(head_size) for better gradient flow
            attention_scores[k] = score / sqrtf((float)head_size);
        }

        // Apply softmax
        softmax(attention_scores, key_seq_length);

        // Compute weighted sum of values
        for (int d = 0; d < head_size; ++d) {
            float weighted_sum = 0.0f;
            for (int k = 0; k < key_seq_length; ++k) {
                int v_idx = k * d_model + head_idx * head_size + d;
                weighted_sum += attention_scores[k] * values[v_idx];
            }
            int out_idx = token_idx * d_model + head_idx * head_size + d;
            output[out_idx] = weighted_sum;
        }
    }
}

Matrix MultiHeadAttention::forward(const Matrix &query, const Matrix &key, const Matrix &value, const Matrix &mask) {
    int seq_length = query.getRows();
    int key_seq_length = key.getRows();
    int d_model = query.getCols();

    Matrix output(seq_length, d_model);

    // Use 2D grid: one dimension for tokens, another for heads
    dim3 blockSize(16, 8);  // 16 tokens, 8 heads per block
    dim3 gridSize((seq_length + blockSize.x - 1) / blockSize.x,
                  (n_heads + blockSize.y - 1) / blockSize.y);
    
    multiHeadAttentionKernel<<<gridSize, blockSize>>>(
        query.getData(),
        key.getData(),
        value.getData(),
        output.getData(),
        d_model,
        n_heads,
        seq_length,
        key_seq_length
    );
    cudaDeviceSynchronize();

    return output;
}

MultiHeadAttention::MultiHeadAttention(size_t d_model, size_t n_heads) 
    : d_model(d_model), n_heads(n_heads) {
    d_k = d_model / n_heads;
    d_v = d_model / n_heads;
    // Inicializar matrices W_Q, W_K, W_V, W_O si las tienes
}

MultiHeadAttention::~MultiHeadAttention() {
    // Liberar memoria si es necesario
    
}