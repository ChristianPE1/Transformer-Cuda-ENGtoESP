#include "multi_head_attention.cuh"
#include <vector>
#include <iostream>
#include <cmath>
#include <cstdlib>

MultiHeadAttention::MultiHeadAttention(int d_model, int num_heads) 
    : d_model(d_model), num_heads(num_heads), d_k(d_model / num_heads), d_v(d_model / num_heads),
      W_q(d_model, d_model), W_k(d_model, d_model), W_v(d_model, d_model), W_o(d_model, d_model) {
    
    // Initialize W_q with Xavier initialization
    std::vector<float> wq_data(d_model * d_model);
    float scale = sqrt(6.0f / (2.0f * d_model));
    for (size_t i = 0; i < wq_data.size(); ++i) {
        wq_data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    }
    W_q.copyFromHost(wq_data);

    // Initialize W_k
    std::vector<float> wk_data(d_model * d_model);
    for (size_t i = 0; i < wk_data.size(); ++i) {
        wk_data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    }
    W_k.copyFromHost(wk_data);
    
    // Initialize W_v
    std::vector<float> wv_data(d_model * d_model);
    for (size_t i = 0; i < wv_data.size(); ++i) {
        wv_data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    }
    W_v.copyFromHost(wv_data);
    
    // Initialize W_o
    std::vector<float> wo_data(d_model * d_model);
    for (size_t i = 0; i < wo_data.size(); ++i) {
        wo_data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    }
    W_o.copyFromHost(wo_data);
    
    std::cout << "[MHA] Initialized with " << num_heads << " heads, d_k=" << d_k << std::endl;
}

Matrix MultiHeadAttention::forward(const Matrix& query, const Matrix& key, const Matrix& value, const Matrix* mask) {
    // Project to Q, K, V
    Matrix Q = query.multiply(W_q);
    Matrix K = key.multiply(W_k);
    Matrix V = value.multiply(W_v);
    
    // Split into multiple heads
    int seq_len = query.getRows();
    
    std::vector<Matrix> attention_outputs;
    attention_outputs.reserve(num_heads);
    
    for (int h = 0; h < num_heads; ++h) {
        // Extract head h from Q, K, V
        Matrix Q_h = extractHead(Q, h, seq_len);
        Matrix K_h = extractHead(K, h, seq_len);
        Matrix V_h = extractHead(V, h, seq_len);
        
        // Compute attention for this head
        Matrix attention_output = computeAttention(Q_h, K_h, V_h, mask);
        attention_outputs.push_back(attention_output);
    }
    
    // Concatenate heads
    Matrix concatenated = combineHeads(attention_outputs);
    
    // Final linear projection
    Matrix output = concatenated.multiply(W_o);
    
    return output;
}

Matrix MultiHeadAttention::extractHead(const Matrix& input, int head_idx, int seq_len) {
    // Extract the portion of the matrix corresponding to head_idx
    Matrix head(seq_len, d_k);
    
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < d_k; ++j) {
            int col_idx = head_idx * d_k + j;
            float val = input.getElement(i, col_idx);
            head.setElement(i, j, val);
        }
    }
    
    return head;
}

Matrix MultiHeadAttention::combineHeads(const std::vector<Matrix>& heads) {
    if (heads.empty()) {
        return Matrix(0, 0);
    }
    
    int seq_len = heads[0].getRows();
    Matrix combined(seq_len, d_model);
    
    for (int h = 0; h < num_heads; ++h) {
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < d_k; ++j) {
                int col_idx = h * d_k + j;
                float val = heads[h].getElement(i, j);
                combined.setElement(i, col_idx, val);
            }
        }
    }
    
    return combined;
}

Matrix MultiHeadAttention::computeAttention(const Matrix& Q, const Matrix& K, const Matrix& V, const Matrix* mask) {
    int seq_len_q = Q.getRows();
    int seq_len_k = K.getRows();
    
    // Compute attention scores: Q * K^T
    Matrix K_transposed = K.transpose();
    Matrix scores(seq_len_q, seq_len_k);
    
    for (int i = 0; i < seq_len_q; ++i) {
        for (int j = 0; j < seq_len_k; ++j) {
            float score = 0.0f;
            for (int k = 0; k < d_k; ++k) {
                score += Q.getElement(i, k) * K_transposed.getElement(k, j);
            }
            // Scale by sqrt(d_k)
            score /= sqrt((float)d_k);
            scores.setElement(i, j, score);
        }
    }
    
    // Apply mask if provided
    if (mask != nullptr) {
        for (int i = 0; i < seq_len_q; ++i) {
            for (int j = 0; j < seq_len_k; ++j) {
                if (mask->getElement(i, j) == 0.0f) {
                    scores.setElement(i, j, -1e9f); // Large negative value
                }
            }
        }
    }
    
    // Apply softmax to get attention weights
    Matrix attention_weights(seq_len_q, seq_len_k);
    for (int i = 0; i < seq_len_q; ++i) {
        // Find max for numerical stability
        float max_score = -1e9f;
        for (int j = 0; j < seq_len_k; ++j) {
            max_score = fmax(max_score, scores.getElement(i, j));
        }
        
        // Compute exp and sum
        float sum_exp = 0.0f;
        for (int j = 0; j < seq_len_k; ++j) {
            float exp_val = exp(scores.getElement(i, j) - max_score);
            attention_weights.setElement(i, j, exp_val);
            sum_exp += exp_val;
        }
        
        // Normalize
        for (int j = 0; j < seq_len_k; ++j) {
            float weight = attention_weights.getElement(i, j) / sum_exp;
            attention_weights.setElement(i, j, weight);
        }
    }
    
    // Apply attention weights to values: attention_weights * V
    Matrix output(seq_len_q, V.getCols());
    for (int i = 0; i < seq_len_q; ++i) {
        for (int j = 0; j < V.getCols(); ++j) {
            float sum = 0.0f;
            for (int k = 0; k < seq_len_k; ++k) {
                sum += attention_weights.getElement(i, k) * V.getElement(k, j);
            }
            output.setElement(i, j, sum);
        }
    }
    
    return output;
}
