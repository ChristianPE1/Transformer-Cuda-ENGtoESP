#ifndef MULTI_HEAD_ATTENTION_CUH
#define MULTI_HEAD_ATTENTION_CUH

#include "../utils/matrix.cuh"
#include <vector>

class MultiHeadAttention {
private:
    size_t d_model;
    size_t num_heads;
    size_t d_k;
    size_t d_v;
    
    // Weight matrices
    Matrix W_q, W_k, W_v, W_o;
    
    // Helper methods
    Matrix computeAttention(const Matrix& Q, const Matrix& K, const Matrix& V, const Matrix* mask = nullptr);
    void splitHeads(const Matrix& input, std::vector<Matrix>& heads);
    Matrix combineHeads(const std::vector<Matrix>& heads);

public:
    MultiHeadAttention(size_t d_model, size_t num_heads = 8);
    
    Matrix forward(const Matrix& query, const Matrix& key, const Matrix& value, const Matrix* mask = nullptr);
    
    // For gradient updates
    void updateWeights(const Matrix& grad_q, const Matrix& grad_k, const Matrix& grad_v, const Matrix& grad_o, float lr);
    
    size_t getDModel() const { return d_model; }
    size_t getNumHeads() const { return num_heads; }
};

#endif // MULTI_HEAD_ATTENTION_CUH
