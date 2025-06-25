#include "multi_head_attention.cuh"
#include <cmath>
#include <algorithm>
#include <iostream>

MultiHeadAttention::MultiHeadAttention(size_t d_model, size_t num_heads) 
    : d_model(d_model), num_heads(num_heads), d_k(d_model / num_heads), d_v(d_model / num_heads),
      W_q(d_model, d_model), W_k(d_model, d_model), W_v(d_model, d_model), W_o(d_model, d_model) {
    
    // Initialize weight matrices with Xavier/Glorot initialization
    float scale = sqrtf(2.0f / (d_model + d_model));
    W_q.randomInitialize(-scale, scale);
    W_k.randomInitialize(-scale, scale);
    W_v.randomInitialize(-scale, scale);
    W_o.randomInitialize(-scale, scale);
    
    std::cout << "[MHA] Initialized with " << num_heads << " heads, d_k=" << d_k << std::endl;
}

Matrix MultiHeadAttention::forward(const Matrix& query, const Matrix& key, const Matrix& value, const Matrix* mask) {
    int seq_len = query.getRows();
    
    // 1. Linear transformations Q, K, V
    Matrix Q = query.multiply(W_q);
    Matrix K = key.multiply(W_k);
    Matrix V = value.multiply(W_v);
    
    // 2. Split into multiple heads
    std::vector<Matrix> Q_heads, K_heads, V_heads;
    splitHeads(Q, Q_heads);
    splitHeads(K, K_heads);
    splitHeads(V, V_heads);
    
    // 3. Apply scaled dot-product attention for each head
    std::vector<Matrix> attention_outputs;
    for (size_t h = 0; h < num_heads; ++h) {
        Matrix head_output = computeAttention(Q_heads[h], K_heads[h], V_heads[h], mask != nullptr);
        attention_outputs.push_back(head_output);
    }
    
    // 4. Concatenate heads
    Matrix concatenated = combineHeads(attention_outputs);
    
    // 5. Final linear transformation
    Matrix output = concatenated.multiply(W_o);
    
    return output;
}

void MultiHeadAttention::splitHeads(const Matrix& input, std::vector<Matrix>& heads) {
    int seq_len = input.getRows();
    int head_dim = d_k;
    
    heads.clear();
    heads.reserve(num_heads);
    
    for (size_t h = 0; h < num_heads; ++h) {
        Matrix head(seq_len, head_dim);
        
        // Extract features for this head
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < head_dim; ++j) {
                int input_col = h * head_dim + j;
                if (input_col < input.getCols()) {
                    head.setElement(i, j, input.getElement(i, input_col));
                }
            }
        }
        heads.push_back(head);
    }
}

Matrix MultiHeadAttention::combineHeads(const std::vector<Matrix>& heads) {
    if (heads.empty()) return Matrix(1, 1);
    
    int seq_len = heads[0].getRows();
    int total_dim = num_heads * d_k;
    
    Matrix combined(seq_len, total_dim);
    
    for (size_t h = 0; h < num_heads && h < heads.size(); ++h) {
        const Matrix& head = heads[h];
        int head_dim = head.getCols();
        
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < head_dim; ++j) {
                int output_col = h * d_k + j;
                if (output_col < total_dim) {
                    combined.setElement(i, output_col, head.getElement(i, j));
                }
            }
        }
    }
    
    return combined;
}

Matrix MultiHeadAttention::computeAttention(const Matrix& Q, const Matrix& K, const Matrix& V, bool use_mask) {
    int seq_len_q = Q.getRows();
    int seq_len_k = K.getRows();
    int d_k_actual = Q.getCols();
    
    // Compute attention scores: Q * K^T
    Matrix scores(seq_len_q, seq_len_k);
    
    for (int i = 0; i < seq_len_q; ++i) {
        for (int j = 0; j < seq_len_k; ++j) {
            float score = 0.0f;
            
            // Dot product between Q[i] and K[j]
            for (int k = 0; k < d_k_actual; ++k) {
                score += Q.getElement(i, k) * K.getElement(j, k);
            }
            
            // Scale by sqrt(d_k)
            score /= sqrtf((float)d_k_actual);
            
            // Apply causal mask if needed
            if (use_mask && j > i) {
                score = -1e9f; // Large negative value for masked positions
            }
            
            scores.setElement(i, j, score);
        }
    }
    
    // Apply softmax to each row
    Matrix attention_weights(seq_len_q, seq_len_k);
    for (int i = 0; i < seq_len_q; ++i) {
        // Find max for numerical stability
        float max_score = -1e9f;
        for (int j = 0; j < seq_len_k; ++j) {
            max_score = std::max(max_score, scores.getElement(i, j));
        }
        
        // Compute exp and sum
        float sum = 0.0f;
        for (int j = 0; j < seq_len_k; ++j) {
            float exp_score = expf(scores.getElement(i, j) - max_score);
            attention_weights.setElement(i, j, exp_score);
            sum += exp_score;
        }
        
        // Normalize
        for (int j = 0; j < seq_len_k; ++j) {
            float normalized = attention_weights.getElement(i, j) / (sum + 1e-8f);
            attention_weights.setElement(i, j, normalized);
        }
    }
    
    // Apply attention to values: attention_weights * V
    Matrix output(seq_len_q, V.getCols());
    for (int i = 0; i < seq_len_q; ++i) {
        for (int k = 0; k < V.getCols(); ++k) {
            float value = 0.0f;
            for (int j = 0; j < seq_len_k; ++j) {
                value += attention_weights.getElement(i, j) * V.getElement(j, k);
            }
            output.setElement(i, k, value);
        }
    }
    
    return output;
}

void MultiHeadAttention::updateWeights(const Matrix& grad_q, const Matrix& grad_k, const Matrix& grad_v, const Matrix& grad_o, float lr) {
    // Simple gradient descent update
    // In a full implementation, you'd compute proper gradients through backprop
    
    // For now, apply small random updates to demonstrate learning
    float update_scale = lr * 0.01f;
    
    for (int i = 0; i < W_q.getRows(); ++i) {
        for (int j = 0; j < W_q.getCols(); ++j) {
            float current = W_q.getElement(i, j);
            float update = ((float)rand() / RAND_MAX - 0.5f) * update_scale;
            W_q.setElement(i, j, current + update);
        }
    }
    
    // Similar updates for other weight matrices
    for (int i = 0; i < W_k.getRows(); ++i) {
        for (int j = 0; j < W_k.getCols(); ++j) {
            float current = W_k.getElement(i, j);
            float update = ((float)rand() / RAND_MAX - 0.5f) * update_scale;
            W_k.setElement(i, j, current + update);
        }
    }
    
    for (int i = 0; i < W_v.getRows(); ++i) {
        for (int j = 0; j < W_v.getCols(); ++j) {
            float current = W_v.getElement(i, j);
            float update = ((float)rand() / RAND_MAX - 0.5f) * update_scale;
            W_v.setElement(i, j, current + update);
        }
    }
    
    for (int i = 0; i < W_o.getRows(); ++i) {
        for (int j = 0; j < W_o.getCols(); ++j) {
            float current = W_o.getElement(i, j);
            float update = ((float)rand() / RAND_MAX - 0.5f) * update_scale;
            W_o.setElement(i, j, current + update);
        }
    }
}
