// filepath: /cuda-transformer/cuda-transformer/src/transformer/transformer.cu
#include "transformer.cuh"
#include "attention.cuh"
#include "embeddings.cuh"
#include "encoder.cuh"
#include "decoder.cuh"
#include "layer_norm.cuh"
#include "feed_forward.cuh"
#include "linear.cuh"
#include "../utils/matrix.cuh"
#include "../utils/cuda_utils.cuh"
#include "../utils/mask_utils.cuh"

__device__ Matrix forward(const Matrix &source, const Matrix &target, Transformer &transformer) {
    Matrix encoder_output = transformer.encode(source);
    Matrix decoder_output = transformer.decode(target, encoder_output);
    return decoder_output;
}

__global__ void transformerKernel(const Matrix source, const Matrix target, Matrix output, Transformer transformer) {
    output = forward(source, target, transformer);
}

Transformer::Transformer(size_t input_vocab_size, size_t target_vocab_size,
                          size_t d_model, size_t n_heads, size_t n_layers, size_t d_ff)
    : d_model(d_model), n_layers(n_layers),
      input_vocab_size(input_vocab_size), target_vocab_size(target_vocab_size),
      pos_encoding(d_model),
      input_embedding(input_vocab_size, d_model),
      target_embedding(target_vocab_size, d_model),
      output_projection(d_model, target_vocab_size) {
    
    for (size_t i = 0; i < n_layers; ++i) {
        encoder_layers.emplace_back(d_model, n_heads, d_ff);
    }

    for (size_t i = 0; i < n_layers; ++i) {
        decoder_layers.emplace_back(d_model, n_heads, d_ff);
    }

    output_projection.initializeXavier();
}

__device__ Matrix Transformer::encode(const Matrix &input_tokens) {
    Matrix embeddings = input_embedding.forward(input_tokens);
    embeddings = embeddings.scale(sqrt(d_model));
    Matrix pos_enc = pos_encoding.getEncoding(input_tokens.getCols());
    Matrix encoder_input = embeddings.add(pos_enc);
    Matrix src_mask = MaskUtils::createPaddingMask(input_tokens);
    
    Matrix output = encoder_input;
    for (auto &layer : encoder_layers) {
        output = layer.forward(output, &src_mask);
    }
    return output;
}

__device__ Matrix Transformer::decode(const Matrix &target_tokens, const Matrix &encoder_output) {
    Matrix embeddings = target_embedding.forward(target_tokens);
    embeddings = embeddings.scale(sqrt(d_model));
    Matrix pos_enc = pos_encoding.getEncoding(target_tokens.getCols());
    Matrix decoder_input = embeddings.add(pos_enc);
    Matrix target_mask = MaskUtils::combineDecoderMasks(target_tokens);
    Matrix src_mask = MaskUtils::createPaddingMask(target_tokens);
    
    Matrix output = decoder_input;
    for (auto &layer : decoder_layers) {
        output = layer.forward(output, encoder_output, target_mask, &src_mask);
    }
    return output;
}

__device__ Matrix Transformer::forward(const Matrix &source_tokens, const Matrix &target_tokens) {
    Matrix encoder_output = encode(source_tokens);
    Matrix decoder_output = decode(target_tokens, encoder_output);
    return decoder_output.multiply(output_projection);
}