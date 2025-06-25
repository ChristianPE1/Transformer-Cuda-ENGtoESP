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

    // Scale embeddings properly (this is important for training stability)
    std::vector<float> embed_data;
    embeddings.copyToHost(embed_data);
    float scale = sqrt((float)d_model);
    for (auto &val : embed_data) {
        val *= scale;
    }
    embeddings.copyFromHost(embed_data);

    // Add positional encoding
    Matrix pos_enc = pos_encoding.getEncoding(input_tokens.size());
    Matrix encoder_input = embeddings.add(pos_enc);

    // Apply simple layer normalization
    encoder_input = applyLayerNorm(encoder_input);

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
    float scale = sqrt((float)d_model);
    for (auto &val : embed_data) {
        val *= scale;
    }
    embeddings.copyFromHost(embed_data);

    // Add positional encoding
    Matrix pos_enc = pos_encoding.getEncoding(target_tokens.size());
    Matrix decoder_input = embeddings.add(pos_enc);

    // Apply layer normalization
    decoder_input = applyLayerNorm(decoder_input);

    // Enhanced cross-attention with residual connections
    Matrix attended_output = applyCrossAttention(decoder_input, encoder_output);
    
    // Residual connection + layer norm
    Matrix decoder_output = decoder_input.add(attended_output);
    decoder_output = applyLayerNorm(decoder_output);

    return decoder_output;
}

// Simple layer normalization implementation
Matrix Transformer::applyLayerNorm(const Matrix& input) {
    int rows = input.getRows();
    int cols = input.getCols();
    Matrix output(rows, cols);
    
    for (int i = 0; i < rows; ++i) {
        // Calculate mean
        float mean = 0.0f;
        for (int j = 0; j < cols; ++j) {
            mean += input.getElement(i, j);
        }
        mean /= cols;
        
        // Calculate variance
        float variance = 0.0f;
        for (int j = 0; j < cols; ++j) {
            float diff = input.getElement(i, j) - mean;
            variance += diff * diff;
        }
        variance /= cols;
        
        // Normalize
        float std_dev = sqrtf(variance + 1e-6f);
        for (int j = 0; j < cols; ++j) {
            float normalized = (input.getElement(i, j) - mean) / std_dev;
            output.setElement(i, j, normalized);
        }
    }
    
    return output;
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
    
    // Encode with improved processing
    Matrix encoder_output = encode(source_tokens);
    std::cout << "[DEBUG] Encode OK - shape: " << encoder_output.getRows() << "x" << encoder_output.getCols() << std::endl;

    // Decode with improved cross-attention
    Matrix decoder_output = decode(target_tokens, encoder_output);
    std::cout << "[DEBUG] Decode OK - shape: " << decoder_output.getRows() << "x" << decoder_output.getCols() << std::endl;

    // Project to vocabulary with improved attention mechanism
    Matrix output(target_tokens.size(), target_vocab_size, 0.0f);
    std::cout << "[DEBUG] Created output matrix: " << output.getRows() << "x" << output.getCols() << std::endl;

    // Improved cross-attention with proper normalization
    for (int i = 0; i < target_tokens.size(); ++i) {
        
        // Calculate cross-attention weights with proper softmax
        std::vector<float> cross_attention(source_tokens.size(), 0.0f);
        float max_score = -1e9f;
          // First pass: calculate raw attention scores with positional bias
        for (int j = 0; j < source_tokens.size(); ++j) {
            float attention_score = 0.0f;
            int context_size = std::min(32, (int)d_model);
            
            for (int d = 0; d < context_size; ++d) {
                float decoder_val = decoder_output.getElement(i, d);
                float encoder_val = encoder_output.getElement(j, d);
                attention_score += decoder_val * encoder_val;
            }
            attention_score /= sqrtf(context_size); // Scale by sqrt(d_k)
            
            // ADD POSITIONAL BIAS to break the tie toward position 0
            float positional_bias = 0.0f;
            if (source_tokens.size() > 1) {
                // Create different preferences for different target positions
                float target_relative = (float)i / std::max(1.0f, (float)(target_tokens.size() - 1));
                float source_relative = (float)j / std::max(1.0f, (float)(source_tokens.size() - 1));
                
                // Diagonal attention pattern (beginning matches beginning, end matches end)
                positional_bias = 2.0f * (1.0f - abs(target_relative - source_relative));
                
                // Add some randomness based on position indices to break symmetry
                positional_bias += sin((float)(i * 7 + j * 11)) * 0.5f;
            }
            
            attention_score += positional_bias;
            cross_attention[j] = attention_score;
            max_score = std::max(max_score, attention_score);
        }
        
        // Second pass: apply softmax
        float attention_sum = 0.0f;
        for (int j = 0; j < source_tokens.size(); ++j) {
            cross_attention[j] = expf(cross_attention[j] - max_score);
            attention_sum += cross_attention[j];
        }
        for (int j = 0; j < source_tokens.size(); ++j) {
            cross_attention[j] /= (attention_sum + 1e-8f);
        }
        
        // Generate vocabulary scores with better context integration
        for (int v = 0; v < target_vocab_size; ++v) { 
            float score = 0.0f;
            
            // 1. Direct similarity with target embedding
            std::vector<int> temp_token = {v};
            Matrix vocab_embedding = target_embedding.forward(temp_token);
            
            float direct_similarity = 0.0f;
            for (int d = 0; d < std::min(64, (int)d_model); ++d) {
                float decoder_val = decoder_output.getElement(i, d);
                float vocab_val = vocab_embedding.getElement(0, d);
                direct_similarity += decoder_val * vocab_val;
            }
            direct_similarity /= std::min(64, (int)d_model);
            score += direct_similarity * 2.0f; // Weight this heavily
            
            // 2. Source context contribution via cross-attention
            float source_context = 0.0f;
            for (int j = 0; j < source_tokens.size(); ++j) {
                // Get source token embedding for context
                std::vector<int> src_token = {source_tokens[j]};
                Matrix src_embedding = input_embedding.forward(src_token);
                
                float context_similarity = 0.0f;
                for (int d = 0; d < std::min(32, (int)d_model); ++d) {
                    float vocab_val = vocab_embedding.getElement(0, d);
                    float src_val = src_embedding.getElement(0, d);
                    context_similarity += vocab_val * src_val;
                }
                source_context += cross_attention[j] * context_similarity;
            }
            score += source_context * 0.5f;
            
            // 3. Position-aware bias (favor shorter, common words early)
            if (i == 0 && v < 100) score += 0.2f; // Boost common words at start
            if (i > 0 && v < 50) score += 0.1f;   // Moderate boost for common words
            
            // 4. Length preference (discourage very short/long sequences)
            int current_len = i + 1;
            int target_len = std::max(2, (int)(source_tokens.size() * 0.8));
            if (current_len < target_len && v != 3) score += 0.05f; // Continue generating
            if (current_len >= target_len && v == 3) score += 1.0f; // Favor EOS at right time
            
            output.setElement(i, v, score);
        }
        
        // Debug attention every few positions
        if (i % 2 == 0) {
            int max_attention_pos = 0;
            for (int j = 1; j < source_tokens.size(); ++j) {
                if (cross_attention[j] > cross_attention[max_attention_pos]) {
                    max_attention_pos = j;
                }
            }
            std::cout << "[DEBUG] Processed row " << i 
                      << " (attending to source pos " << max_attention_pos 
                      << " with weight " << std::fixed << std::setprecision(2) << cross_attention[max_attention_pos] << ")" << std::endl;
        }
    }
    
    std::cout << "[DEBUG] Forward completed!" << std::endl;
    return output;
}

std::vector<int> Transformer::generate(const std::vector<int> &source_tokens,
int sos_token, int eos_token, size_t max_length)
{
    std::vector<int> generated = {sos_token};
    
    // Estimate target length more conservatively
    size_t target_length = std::max(2, (int)(source_tokens.size() * 0.8));
    size_t actual_max = std::min(max_length, target_length + 3);

    for (size_t step = 0; step < actual_max; ++step)
    {
        Matrix output = forward(source_tokens, generated);

        // Get predictions for the last position
        int last_pos = generated.size() - 1;
        
        // Collect candidates with better scoring
        std::vector<std::pair<float, int>> candidates;
        
        for (int v = 0; v < target_vocab_size; ++v)
        {
            float score = output.getElement(last_pos, v);
            
            // Enhanced filtering and scoring
            
            // 1. Strongly discourage SOS repetition
            if (v == sos_token && generated.size() > 1) {
                score -= 20.0f;
                continue;
            }
            
            // 2. Context-aware EOS timing
            if (v == eos_token) {
                if (generated.size() >= target_length) {
                    score += 8.0f; // Strong boost when we should end
                } else if (generated.size() < 2) {
                    score -= 15.0f; // Discourage very early ending
                }
            }
              // 3. Prevent immediate repetition of last 2 tokens
            for (int i = std::max(1, (int)generated.size() - 2); i < generated.size(); i++) {
                if (generated[i] == v) {
                    score -= 5.0f; // Penalize recent repetitions
                    break;
                }
            }
            
            // 4. Boost common words early, rare words later
            if (step < 2) {
                if (v < 100) score += 0.3f; // Common words early
            } else {
                if (v >= 100 && v < 500) score += 0.1f; // Mid-frequency words later
            }
            
            // 5. Length-based adjustments
            if (generated.size() > target_length + 1 && v != eos_token) {
                score -= 3.0f; // Discourage continuing too long
            }
            
            candidates.push_back({score, v});
        }
        
        // Sort by score
        std::sort(candidates.begin(), candidates.end(), std::greater<std::pair<float, int>>());
        
        // Improved token selection
        int best_token = candidates[0].second;
        float best_score = candidates[0].first;
        
        // Use temperature-based sampling for first few tokens for diversity
        if (step < 2 && candidates.size() > 3) {
            float temperature = 0.8f;
            std::vector<float> probs;
            float max_score = candidates[0].first;
            float sum = 0.0f;
            
            // Calculate probabilities for top candidates
            for (int i = 0; i < std::min(5, (int)candidates.size()); ++i) {
                float prob = expf((candidates[i].first - max_score) / temperature);
                probs.push_back(prob);
                sum += prob;
            }
            
            // Normalize
            for (float& p : probs) p /= sum;
            
            // Sample from top 3
            float rand_val = ((float)rand() / RAND_MAX);
            float cumsum = 0.0f;
            for (int i = 0; i < std::min(3, (int)probs.size()); ++i) {
                cumsum += probs[i];
                if (rand_val <= cumsum) {
                    best_token = candidates[i].second;
                    best_score = candidates[i].first;
                    break;
                }
            }
        }

        // Enhanced debug output
        if (step < 3) {
            std::cout << "[GEN] Step " << step << " - Best token: " << best_token 
                      << " (score: " << std::fixed << std::setprecision(1) << best_score 
                      << ", target_len: " << target_length << ")";
            
            std::cout << " [Top scores: ";
            for (int i = 0; i < std::min(5, (int)candidates.size()); ++i) {
                std::cout << candidates[i].second << ":" << std::fixed << std::setprecision(1) << candidates[i].first << " ";
            }
            std::cout << "]" << std::endl;
        }

        generated.push_back(best_token);

        // Stop on EOS
        if (best_token == eos_token) {
            break;
        }
        
        // Force termination if too long
        if (generated.size() >= target_length + 2) {
            if (generated.back() != eos_token) {
                generated.push_back(eos_token);
            }
            break;
        }
    }
    
    // Ensure EOS ending
    if (generated.back() != eos_token && generated.size() < max_length) {
        generated.push_back(eos_token);
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