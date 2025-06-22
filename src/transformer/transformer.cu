#include "transformer.cuh"
#include "embeddings.cuh"
#include "../utils/matrix.cuh"
#include <iostream>
#include <cmath>

Transformer::Transformer(size_t input_vocab_size, size_t target_vocab_size,size_t d_model, size_t n_heads, size_t n_layers, size_t d_ff)
    : input_vocab_size(input_vocab_size), target_vocab_size(target_vocab_size),
      d_model(d_model), n_layers(n_layers),
      input_embedding(input_vocab_size, d_model),
      target_embedding(target_vocab_size, d_model),
      pos_encoding(d_model)
{

    std::cout << "Transformer initialized:" << std::endl;
    std::cout << "  Input vocab: " << input_vocab_size << std::endl;
    std::cout << "  Target vocab: " << target_vocab_size << std::endl;
    std::cout << "  d_model: " << d_model << std::endl;
    std::cout << "  layers: " << n_layers << std::endl;
}

Matrix Transformer::encode(const std::vector<int> &input_tokens)
{
    // Get embeddings
    Matrix embeddings = input_embedding.forward(input_tokens);

    // Scale embeddings
    std::vector<float> embed_data;
    embeddings.copyToHost(embed_data);
    float scale = sqrt(d_model);
    for (auto &val : embed_data)
    {
        val *= scale;
    }
    embeddings.copyFromHost(embed_data);

    // Add positional encoding
    Matrix pos_enc = pos_encoding.getEncoding(input_tokens.size());
    Matrix encoder_input = embeddings.add(pos_enc);

    // For now, return encoder_input (no actual encoder layers yet)
    return encoder_input;
}

Matrix Transformer::decode(const std::vector<int> &target_tokens,
                           const Matrix &encoder_output)
{
    // Get target embeddings
    Matrix embeddings = target_embedding.forward(target_tokens);

    // Scale embeddings
    std::vector<float> embed_data;
    embeddings.copyToHost(embed_data);
    float scale = sqrt(d_model);
    for (auto &val : embed_data)
    {
        val *= scale;
    }
    embeddings.copyFromHost(embed_data);

    // Add positional encoding
    Matrix pos_enc = pos_encoding.getEncoding(target_tokens.size());
    Matrix decoder_input = embeddings.add(pos_enc);

    // For now, return decoder_input (no actual decoder layers yet)
    return decoder_input;
}

Matrix Transformer::forward(const std::vector<int> &source_tokens,
                            const std::vector<int> &target_tokens)
{
    // Encode
    Matrix encoder_output = encode(source_tokens);

    // Decode
    Matrix decoder_output = decode(target_tokens, encoder_output);

    // Project to vocabulary (simplified linear projection)
    Matrix output(target_tokens.size(), target_vocab_size, 0.0f);

    // Simple projection (this should be a proper linear layer)
    std::vector<float> decoder_data;
    decoder_output.copyToHost(decoder_data);

    for (int i = 0; i < target_tokens.size(); ++i)
    {
        for (int v = 0; v < target_vocab_size; ++v)
        {
            float sum = 0.0f;
            for (int j = 0; j < d_model; ++j)
            {
                if (i * d_model + j < decoder_data.size())
                {
                    sum += decoder_data[i * d_model + j] * (0.01f * (v + j + 1));
                }
            }
            output.setElement(i, v, sum);
        }
    }

    return output;
}

std::vector<int> Transformer::generate(const std::vector<int> &source_tokens,
int sos_token, int eos_token, size_t max_length)
{
    std::vector<int> generated = {sos_token};

    for (size_t step = 0; step < max_length; ++step)
    {
        Matrix output = forward(source_tokens, generated);

        // Get last token predictions
        int last_pos = generated.size() - 1;
        int best_token = 0;
        float best_score = output.getElement(last_pos, 0);

        for (int v = 1; v < target_vocab_size; ++v)
        {
            float score = output.getElement(last_pos, v);
            if (score > best_score)
            {
                best_score = score;
                best_token = v;
            }
        }

        generated.push_back(best_token);

        if (best_token == eos_token)
        {
            break;
        }
    }

    return generated;
}