// filepath: cuda-transformer/cuda-transformer/src/transformer/transformer.cuh
#ifndef TRANSFORMER_CUH
#define TRANSFORMER_CUH

#include <vector>
#include "attention.cuh"
#include "embeddings.cuh"
#include "encoder.cuh"
#include "decoder.cuh"
#include "layer_norm.cuh"
#include "feed_forward.cuh"
#include "linear.cuh"
#include "mask_utils.cuh"
#include "matrix.cuh"

class Transformer {
private:
    std::vector<EncoderLayer> encoder_layers;
    std::vector<DecoderLayer> decoder_layers;
    PositionalEncoding pos_encoding;
    Embedding input_embedding;
    Embedding target_embedding;
    Linear output_projection; // Linear layer final para vocabulario de salida
    size_t d_model;
    size_t n_layers;
    size_t input_vocab_size;
    size_t target_vocab_size;

public:
    Transformer(size_t input_vocab_size, size_t target_vocab_size,
                size_t d_model = 512, size_t n_heads = 8,
                size_t n_layers = 6, size_t d_ff = 2048);

    Matrix encode(const std::vector<int>& input_tokens);
    Matrix decode(const std::vector<int>& target_tokens, const Matrix& encoder_output,
                  const std::vector<int>& input_tokens);
    Matrix forward(const std::vector<int>& source_tokens, const std::vector<int>& target_tokens);
    std::vector<int> generate(const std::vector<int>& source_tokens,
                              int sos_token = 1, int eos_token = 2,
                              size_t max_length = 50);
};

#endif // TRANSFORMER_CUH