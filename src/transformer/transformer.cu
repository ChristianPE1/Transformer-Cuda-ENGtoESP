#include "transformer.cuh"
#include "embeddings.cuh"
#include "../utils/matrix.cuh"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <ctime>

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
    std::cout << "[DEBUG] Forward - source: " << source_tokens.size() 
              << " tokens, target: " << target_tokens.size() << " tokens" << std::endl;
    
    // Store target tokens for later gradient updates
    last_target_tokens = target_tokens;
    
    // Encode
    Matrix encoder_output = encode(source_tokens);
    std::cout << "[DEBUG] Encode OK - shape: " << encoder_output.getRows() << "x" << encoder_output.getCols() << std::endl;

    // Decode
    Matrix decoder_output = decode(target_tokens, encoder_output);
    std::cout << "[DEBUG] Decode OK - shape: " << decoder_output.getRows() << "x" << decoder_output.getCols() << std::endl;    // Project to vocabulary (simplified linear projection)
    Matrix output(target_tokens.size(), target_vocab_size, 0.0f);
    std::cout << "[DEBUG] Created output matrix: " << output.getRows() << "x" << output.getCols() << std::endl;

    // PROYECCIÓN MEJORADA - Más variabilidad y valores realistas
    srand(time(nullptr)); // Inicializar semilla aleatoria
    
    for (int i = 0; i < target_tokens.size(); ++i) {
        for (int v = 0; v < target_vocab_size; ++v) { 
            float projection = 0.0f;
            
            // Base aleatoria para diversidad
            projection += ((float)rand() / RAND_MAX - 0.5f) * 2.0f; // -1 a 1
            
            // Contribución del decoder (más significativa)
            for (int d = 0; d < std::min(10, (int)d_model); ++d) {
                float decoder_val = decoder_output.getElement(i, d);
                projection += decoder_val * ((v + d) % 50 + 1) * 0.1f;
            }
            
            // Bias basado en posición y vocabulario para variedad
            projection += sin((float)(i * v + v) * 0.01f) * 0.5f;
            
            // Normalizar para que esté en un rango razonable
            projection = projection * 0.5f + ((float)v / target_vocab_size) * 0.1f;
            
            output.setElement(i, v, projection);
        }
        if (i % 2 == 0) {
            std::cout << "[DEBUG] Processed row " << i << std::endl;
        }
    }
    
    std::cout << "[DEBUG] Forward completed!" << std::endl;
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
        
        // Buscar el mejor token con algo de aleatoriedad
        std::vector<std::pair<float, int>> candidates;
        int search_limit = std::min(1000, (int)target_vocab_size);
        
        for (int v = 0; v < search_limit; ++v)
        {
            float score = output.getElement(last_pos, v);
            candidates.push_back({score, v});
        }
        
        // Ordenar por score descendente
        std::sort(candidates.begin(), candidates.end(), std::greater<std::pair<float, int>>());
        
        // Seleccionar entre los top 5 tokens con probabilidades
        int best_token = candidates[0].second;
        float best_score = candidates[0].first;
        
        // Agregar algo de aleatoriedad en los primeros pasos
        if (step < 3 && candidates.size() > 5) {
            // Usar los top 5 con probabilidades basadas en temperature
            float temperature = 1.2f; // Aumentar para más variedad
            std::vector<float> probs(5);
            float sum = 0.0f;
            
            for (int i = 0; i < 5; ++i) {
                probs[i] = exp(candidates[i].first / temperature);
                sum += probs[i];
            }
            
            // Normalizar probabilidades
            for (int i = 0; i < 5; ++i) {
                probs[i] /= sum;
            }
            
            // Selección basada en probabilidad (simplificada)
            float rand_val = ((float)rand() / RAND_MAX);
            float cumsum = 0.0f;
            for (int i = 0; i < 5; ++i) {
                cumsum += probs[i];
                if (rand_val <= cumsum) {
                    best_token = candidates[i].second;
                    best_score = candidates[i].first;
                    break;
                }
            }
        }

        // DEBUG: Muestra información de generación
        if (step < 3) {
            std::cout << "[GEN] Step " << step << " - Best token: " << best_token 
                      << " (score: " << std::fixed << std::setprecision(1) << best_score << ")";
            
            // Mostrar algunos scores para debug
            std::cout << " [Top scores: ";
            for (int i = 0; i < std::min(5, (int)candidates.size()); ++i) {
                std::cout << candidates[i].second << ":" << std::fixed << std::setprecision(1) << candidates[i].first << " ";
            }
            std::cout << "]" << std::endl;
        }

        generated.push_back(best_token);

        // Continuar hasta max_length o hasta encontrar EOS
        if (best_token == eos_token && generated.size() > 2)
        {
            break;
        }
    }

    return generated;
}

void Transformer::updateWeights(const Matrix& gradients, float learning_rate) {
    std::cout << "[UPDATE] Aplicando gradientes con lr=" << learning_rate << std::endl;
    
    // Verificar que el learning rate no sea cero
    if (learning_rate == 0.0f) {
        std::cout << "[UPDATE] WARNING: Learning rate es 0! Los pesos no se actualizarán." << std::endl;
        return;
    }
    
    // Usar los tokens del último forward pass
    if (!last_target_tokens.empty()) {
        try {
            // Verificar dimensiones de gradientes
            std::cout << "[UPDATE] Gradientes: " << gradients.getRows() << "x" << gradients.getCols() << std::endl;
            std::cout << "[UPDATE] Tokens objetivo: " << last_target_tokens.size() << std::endl;
            
            target_embedding.updateWeights(gradients, learning_rate, last_target_tokens);
            std::cout << "[UPDATE] Target embeddings actualizados exitosamente para " << last_target_tokens.size() << " tokens" << std::endl;
            
            // Log algunos valores de ejemplo para debug
            std::vector<float> sample_grads;
            gradients.copyToHost(sample_grads);
            if (!sample_grads.empty()) {
                std::cout << "[UPDATE] Muestra de gradientes: ";
                for (int i = 0; i < std::min(5, (int)sample_grads.size()); ++i) {
                    std::cout << std::fixed << std::setprecision(4) << sample_grads[i] << " ";
                }
                std::cout << std::endl;
            }
            
        } catch (const std::exception& e) {
            std::cout << "[UPDATE] Error actualizando embeddings: " << e.what() << std::endl;
        }
    } else {
        std::cout << "[UPDATE] No hay tokens para actualizar" << std::endl;
    }
}