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
        // Inicialización actual
        // proj_data[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;

        // Cambiar por (valores más grandes)
        proj_data[i] = ((float)rand() / RAND_MAX - 0.5f) * 1.0f; // 10x más grande
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

    // Añadir bias más variado y basado en posición
    for (int i = 0; i < logits.getRows(); i++) {
        // Bias diferente para cada posición y más tokens
        int pos_bias = (i + 1) % 10; // Varía según posición
        
        logits.setElement(i, 50 + pos_bias, logits.getElement(i, 50 + pos_bias) + 3.0f);
        logits.setElement(i, 100 + pos_bias, logits.getElement(i, 100 + pos_bias) + 2.5f);
        logits.setElement(i, 200 + pos_bias, logits.getElement(i, 200 + pos_bias) + 2.0f);
        logits.setElement(i, 300 + pos_bias, logits.getElement(i, 300 + pos_bias) + 1.5f);
        
        // Agregar algo de aleatoriedad basada en step
        if (i > 0) {
            int rand_token = (i * 37 + 123) % 500 + 10; // Pseudo-aleatorio
            logits.setElement(i, rand_token, logits.getElement(i, rand_token) + 1.0f);
        }
    }

    // DEBUG: Verificar logits antes del softmax
    std::cout << "[DEBUG] Logits antes de softmax (fila 0, primeros 10): ";
    for (int v = 0; v < 10; ++v) {
        std::cout << std::fixed << std::setprecision(2) << logits.getElement(0, v) << " ";
    }
    std::cout << std::endl;

    // Softmax en GPU
    Matrix output = logits.softmax();
    
    // DEBUG: Verificar output después del softmax
    std::cout << "[DEBUG] Output después de softmax (fila 0, primeros 10): ";
    for (int v = 0; v < 10; ++v) {
        std::cout << std::fixed << std::setprecision(4) << output.getElement(0, v) << " ";
    }
    std::cout << std::endl;
    
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
        
        // Solo busca en los primeros 1000 tokens y incluye los tokens con bias
        for (int v = 1; v < std::min(1000, (int)target_vocab_size); ++v) {
            float score = output.getElement(last_pos, v);
            if (score > best_score) {
                best_score = score;
                best_token = v;
            }
        }

        // DEBUG: Muestra información más detallada
        if (step < 3) {
            std::cout << "[GEN] Step " << step << " - Token: " << best_token 
                      << " (score: " << std::fixed << std::setprecision(3) << best_score;
            
            // Mostrar algunos scores más
            std::cout << ") - Top scores: ";
            for (int v = 0; v < 5; ++v) {
                std::cout << v << ":" << std::fixed << std::setprecision(3) 
                          << output.getElement(last_pos, v) << " ";
            }
            std::cout << std::endl;
        }

        generated.push_back(best_token);

        // DEBUG: Mostrar estado de generación
        if (step < 5) {
            std::cout << "[GEN] Generated so far: ";
            for (int t : generated) std::cout << t << " ";
            std::cout << "(eos=" << eos_token << ")" << std::endl;
        }

        // Solo parar si encuentra EOS y ha generado al menos 3 tokens
        if (best_token == eos_token && generated.size() > 3) {
            std::cout << "[GEN] Stopping: Found EOS token" << std::endl;
            break;
        }
        
        // También parar si genera muchos tokens 0 seguidos (puede ser un bucle)
        if (best_token == 0 && step > 1) {
            std::cout << "[GEN] Warning: Generated token 0, continuing..." << std::endl;
        }
    }

    // Añade este debug
    std::cout << "[DEBUG] Checking projection_weights:" << std::endl;
    for (int i = 0; i < std::min(5, (int)d_model); i++) {
        for (int j = 0; j < std::min(5, (int)target_vocab_size); j++) {
            std::cout << projection_weights.getElement(i, j) << " ";
        }
        std::cout << std::endl;
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