#include "transformer.cuh"
#include "embeddings.cuh"
#include "../utils/matrix.cuh"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>    // Para std::max_element, std::min_element
#include <vector>       // Para std::vector
#include <numeric>      // Para std::accumulate
#include <functional>   // Para std::function

Transformer::Transformer(size_t input_vocab_size, size_t target_vocab_size,
    size_t d_model, size_t n_heads, size_t n_layers, size_t d_ff)
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
    // Encode y Decode (ya funcionan en GPU)
    Matrix encoder_output = encode(source_tokens);
    Matrix decoder_output = decode(target_tokens, encoder_output);

    // PROYECCIÓN OPTIMIZADA EN GPU
    Matrix projection_weights(d_model, target_vocab_size, 0.01f);  // Pesos dummy por ahora
    
    // Multiplicación de matrices en GPU (no en CPU)
    Matrix logits = decoder_output.matrixMultiply(projection_weights);
    
    // Softmax en GPU (no en CPU)
    Matrix output = logits.softmax();
    
    return output;
}

std::vector<int> Transformer::generate(const std::vector<int> &source_tokens,
int sos_token, int eos_token, size_t max_length)
{
    std::vector<int> generated = {sos_token};

    for (size_t step = 0; step < max_length; ++step)
    {
        Matrix output = forward(source_tokens, generated);
        
        // OPTIMIZACIÓN: Copia solo la última fila de una vez
        int last_pos = generated.size() - 1;
        std::vector<float> last_row_data;
        output.copyToHostBatch(last_row_data);
        
        // Busca el mejor token en CPU (datos ya copiados)
        int best_token = 0;
        float best_score = last_row_data[last_pos * target_vocab_size + 0];
        
        for (int v = 1; v < std::min(1000, (int)target_vocab_size); ++v) {
            float score = last_row_data[last_pos * target_vocab_size + v];
            if (score > best_score) {
                best_score = score;
                best_token = v;
            }
        }

        generated.push_back(best_token);

        if (best_token == eos_token && generated.size() > 2) {
            break;
        }
    }

    return generated;
}

void Transformer::updateWeights(const Matrix& gradients, float learning_rate) {
    //std::cout << "[UPDATE] Aplicando gradientes con lr=" << learning_rate << std::endl;
    
    // Usar los tokens del último forward pass
    if (!last_target_tokens.empty()) {
        try {
            target_embedding.updateWeights(gradients, learning_rate, last_target_tokens);
            //std::cout << "[UPDATE] Target embeddings actualizados para " << last_target_tokens.size() << " tokens" << std::endl;
        } catch (const std::exception& e) {
            //std::cout << "[UPDATE] Error actualizando embeddings: " << e.what() << std::endl;
        }
    } else {
        //std::cout << "[UPDATE] No hay tokens para actualizar" << std::endl;
    }
}