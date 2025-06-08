// filepath: cuda-transformer/cuda-transformer/src/transformer/encoder.cuh
#ifndef ENCODER_H
#define ENCODER_H

#include "common.cuh"
#include "attention.cuh"
#include "feed_forward.cuh"
#include "layer_norm.cuh"

class EncoderLayer {
private:
    MultiHeadAttention self_attention;
    FeedForward feed_forward;
    LayerNorm norm1, norm2;

public:
    EncoderLayer(size_t d_model, size_t n_heads, size_t d_ff = 2048)
        : self_attention(d_model, n_heads), 
          feed_forward(d_model, d_ff),
          norm1(d_model), 
          norm2(d_model) {}

    __device__ Matrix forward(const Matrix &input, const Matrix *src_mask = nullptr) {
        // Masked self-attention
        Matrix self_att_output = self_attention.forward(input, input, input, src_mask);
        Matrix norm1_output = norm1.forward(input.add(self_att_output));

        // Feed-forward
        Matrix ff_output = feed_forward.forward(norm1_output);
        Matrix norm2_output = norm2.forward(norm1_output.add(ff_output));

        return norm2_output;
    }
};

#endif // ENCODER_H