#ifndef TRANSFORMER_CUH
#define TRANSFORMER_CUH

#include <vector>
#include "embeddings.cuh"
#include "../utils/matrix.cuh"
#include "../layers/multi_head_attention.cuh"
#include "../layers/feed_forward.cuh"

class Transformer
{
private:
    Embedding input_embedding;
    Embedding target_embedding;
    PositionalEncoding pos_encoding;
    size_t d_model;
    size_t n_layers;
    size_t n_heads;
    size_t d_ff;
    size_t input_vocab_size;
    size_t target_vocab_size;
    
    // Multi-layer components
    std::vector<MultiHeadAttention*> encoder_self_attention;
    std::vector<FeedForward*> encoder_ffn;
    std::vector<MultiHeadAttention*> decoder_self_attention;
    std::vector<MultiHeadAttention*> decoder_cross_attention;
    std::vector<FeedForward*> decoder_ffn;
    
    // Store tokens from last forward pass for gradient updates
    std::vector<int> last_target_tokens;
    std::vector<int> last_source_tokens;
    
    // Helper methods
    Matrix applyLayerNorm(const Matrix& input);
    Matrix applyEncoderLayer(const Matrix& input, int layer_idx);
    Matrix applyDecoderLayer(const Matrix& input, const Matrix& encoder_output, int layer_idx);

public:
    Transformer(size_t input_vocab_size, size_t target_vocab_size,
                size_t d_model = 512, size_t n_heads = 8,
                size_t n_layers = 6, size_t d_ff = 2048);
    ~Transformer();

    Matrix encode(const std::vector<int> &input_tokens);
    Matrix decode(const std::vector<int> &target_tokens, const Matrix &encoder_output);
    Matrix forward(const std::vector<int> &source_tokens, const std::vector<int> &target_tokens);
    std::vector<int> generate(const std::vector<int> &source_tokens,
                              int sos_token = 2, int eos_token = 3,
                              size_t max_length = 50);
    
    // Training methods
    void updateWeights(const Matrix& gradients, float learning_rate);
    
    size_t getTargetVocabSize() const { return target_vocab_size; }
};

#endif // TRANSFORMER_CUH