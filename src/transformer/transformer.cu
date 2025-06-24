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

    // ATENCIÓN CRUZADA SIMPLE - Mezclar decoder input con encoder output
    Matrix decoder_output = applyCrossAttention(decoder_input, encoder_output);

    return decoder_output;
}

// Nueva función de atención cruzada simple
Matrix Transformer::applyCrossAttention(const Matrix& decoder_input, const Matrix& encoder_output) {
    int decoder_len = decoder_input.getRows();
    int encoder_len = encoder_output.getRows();
    int d_model = decoder_input.getCols();
    
    Matrix attended_output(decoder_len, d_model, 0.0f);
    
    for (int i = 0; i < decoder_len; ++i) {
        for (int d = 0; d < d_model; ++d) {
            float attended_value = 0.0f;
            float attention_sum = 0.0f;
            
            // Calcular atención entre posición i del decoder y todas las del encoder
            for (int j = 0; j < encoder_len; ++j) {
                // Peso de atención simple basado en producto punto
                float attention_score = 0.0f;
                for (int k = 0; k < std::min(16, d_model); ++k) {
                    attention_score += decoder_input.getElement(i, k) * encoder_output.getElement(j, k);
                }
                
                // Normalizar y aplicar softmax simple
                attention_score = exp(attention_score * 0.1f); // Temperature para suavizar
                attention_sum += attention_score;
                
                // Agregar contribución ponderada del encoder
                attended_value += attention_score * encoder_output.getElement(j, d);
            }
            
            // Normalizar por la suma de pesos de atención
            if (attention_sum > 0) {
                attended_value /= attention_sum;
            }
            
            // Combinar con input original (conexión residual)
            float original_value = decoder_input.getElement(i, d);
            float final_value = 0.7f * original_value + 0.3f * attended_value;
            
            attended_output.setElement(i, d, final_value);
        }
    }
    
    return attended_output;
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
    std::cout << "[DEBUG] Decode OK - shape: " << decoder_output.getRows() << "x" << decoder_output.getCols() << std::endl;    // Project to vocabulary usando similitud con embeddings objetivo
    Matrix output(target_tokens.size(), target_vocab_size, 0.0f);
    std::cout << "[DEBUG] Created output matrix: " << output.getRows() << "x" << output.getCols() << std::endl;

    // PROYECCIÓN BASADA EN SIMILITUD CON EMBEDDINGS
    for (int i = 0; i < target_tokens.size(); ++i) {
        for (int v = 0; v < target_vocab_size; ++v) { 
            float similarity = 0.0f;
            
            // Calcular similitud entre decoder output y embedding del token v
            std::vector<int> temp_token = {v};
            Matrix vocab_embedding = target_embedding.forward(temp_token);
            
            // Producto punto normalizado (similitud coseno simplificada)
            for (int d = 0; d < std::min(32, (int)d_model); ++d) {
                float decoder_val = decoder_output.getElement(i, d);
                float vocab_val = vocab_embedding.getElement(0, d);
                similarity += decoder_val * vocab_val;
            }
            
            // Normalizar por dimensión
            similarity /= std::min(32, (int)d_model);
            
            // Agregar bias basado en frecuencia (tokens más comunes tienen mayor probabilidad)
            float frequency_bias = 0.0f;
            if (v < 100) frequency_bias = 0.1f;      // Tokens comunes
            else if (v < 500) frequency_bias = 0.05f; // Tokens moderados
            else frequency_bias = 0.01f;              // Tokens raros
            
            similarity += frequency_bias;
            
            // Pequeña aleatoriedad para diversidad
            similarity += ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
            
            output.setElement(i, v, similarity);
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
          // Seleccionar entre los top tokens con mejor estrategia
        int best_token = candidates[0].second;
        float best_score = candidates[0].first;
        
        // MEJORA: Usar temperatura variable y beam search simple
        if (step < 5 && candidates.size() > 10) {
            // Temperatura más alta al principio para más exploración
            float temperature = 1.5f - (step * 0.2f); // Decrece de 1.5 a 0.5
            
            // Aplicar softmax con temperatura
            std::vector<float> probs(10);
            float sum = 0.0f;
            
            for (int i = 0; i < 10; ++i) {
                probs[i] = exp(candidates[i].first / temperature);
                sum += probs[i];
            }
            
            // Normalizar
            for (int i = 0; i < 10; ++i) {
                probs[i] /= sum;
            }
            
            // Selección ponderada entre top 3 candidatos
            float rand_val = ((float)rand() / RAND_MAX);
            float cumsum = 0.0f;
            for (int i = 0; i < 3; ++i) { // Solo top 3 para ser más conservador
                cumsum += probs[i];
                if (rand_val <= cumsum) {
                    best_token = candidates[i].second;
                    best_score = candidates[i].first;
                    break;
                }
            }
        } else {
            // En pasos posteriores, ser más conservador
            if (candidates.size() > 3) {
                // Elegir aleatoriamente entre top 3
                int choice = rand() % 3;
                best_token = candidates[choice].second;
                best_score = candidates[choice].first;
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