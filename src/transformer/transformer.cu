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
      pos_encoding(d_model),
      projection_weights(d_model, target_vocab_size)  // ← Inicializar pesos reales
{

    // Inicializar pesos de proyección con valores aleatorios
    std::vector<float> proj_data(d_model * target_vocab_size);
    for (size_t i = 0; i < proj_data.size(); ++i) {
        proj_data[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    projection_weights.copyFromHost(proj_data);
    
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
    // Guardar tokens para actualización de pesos
    last_target_tokens = target_tokens;
    
    // Encode y Decode
    Matrix encoder_output = encode(source_tokens);
    Matrix decoder_output = decode(target_tokens, encoder_output);

    // USAR PESOS REALES DE PROYECCIÓN
    Matrix logits = decoder_output.matrixMultiply(projection_weights);
    
    // Softmax en GPU
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
        
        // ARREGLO: Acceso correcto a la última fila
        int last_pos = generated.size() - 1;
        
        // Busca el mejor token directamente en la matriz
        int best_token = 0;
        float best_score = output.getElement(last_pos, 0);
        
        // Solo busca en los primeros 100 tokens para ser rápido
        for (int v = 1; v < std::min(100, (int)target_vocab_size); ++v) {
            float score = output.getElement(last_pos, v);
            if (score > best_score) {
                best_score = score;
                best_token = v;
            }
        }

        // DEBUG: Muestra información
        if (step < 3) {
            std::cout << "[GEN] Step " << step << " - Token: " << best_token 
                      << " (score: " << best_score << ")" << std::endl;
        }

        generated.push_back(best_token);

        if (best_token == eos_token && generated.size() > 2) {
            break;
        }
    }

    return generated;
}

void Transformer::updateWeights(const Matrix& gradients, float learning_rate) {
    std::cout << "[UPDATE] Actualizando pesos con lr=" << learning_rate << std::endl;
    
    try {
        // Actualizar embeddings
        if (!last_target_tokens.empty()) {
            target_embedding.updateWeights(gradients, learning_rate, last_target_tokens);
        }
        
        // Actualizar pesos de proyección (simplificado)
        std::vector<float> proj_data, grad_data;
        projection_weights.copyToHostBatch(proj_data);
        gradients.copyToHostBatch(grad_data);
        
        // Actualización simple: W = W - lr * grad (tamaños compatibles)
        for (size_t i = 0; i < std::min(proj_data.size(), grad_data.size()); ++i) {
            proj_data[i] -= learning_rate * grad_data[i] * 0.1f; // Factor pequeño
        }
        
        projection_weights.copyFromHostBatch(proj_data);
        std::cout << "[UPDATE] Projection weights actualizados" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "[UPDATE] Error: " << e.what() << std::endl;
    }
}