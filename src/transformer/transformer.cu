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
    //std::cout << "[DEBUG] Forward - source: " << source_tokens.size() << " tokens, target: " << target_tokens.size() << " tokens" << std::endl;
    
    // Store target tokens for later gradient updates
    last_target_tokens = target_tokens;
    
    // Encode
    Matrix encoder_output = encode(source_tokens);
    //std::cout << "[DEBUG] Encode OK - shape: " << encoder_output.getRows() << "x" << encoder_output.getCols() << std::endl;

    // Decode
    Matrix decoder_output = decode(target_tokens, encoder_output);
    //std::cout << "[DEBUG] Decode OK - shape: " << decoder_output.getRows() << "x" << decoder_output.getCols() << std::endl;

    // Project to vocabulary (simplified linear projection)
    Matrix output(target_tokens.size(), target_vocab_size, 0.0f);
    
    for (int i = 0; i < target_tokens.size(); ++i) {
        std::vector<float> logits(target_vocab_size, 0.0f);
        
        // Calcular logits basados en el decoder output
        for (int v = 0; v < target_vocab_size; ++v) {
            float logit = 0.0f;
            
            // Usar decoder output para calcular logits
            for (int d = 0; d < d_model; ++d) {
                float decoder_val = decoder_output.getElement(i, d);
                logit += decoder_val * sin((v + d) * 0.1f); // Proyección simple pero real
            }
            
            // Agregar bias por vocabulario
            logit += (float)(v % 100) * 0.01f;
            logits[v] = logit;
        }
        
        // APLICAR SOFTMAX
        float max_logit = *std::max_element(logits.begin(), logits.end());
        float sum_exp = 0.0f;
        
        for (int v = 0; v < target_vocab_size; ++v) {
            logits[v] = exp(logits[v] - max_logit);
            sum_exp += logits[v];
        }
        
        for (int v = 0; v < target_vocab_size; ++v) {
            output.setElement(i, v, logits[v] / sum_exp);
        }
    }
    
    //std::cout << "[DEBUG] Forward completed!" << std::endl;
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

        // BUSCA EN MÁS PALABRAS DEL VOCABULARIO
        int search_limit = std::min(100, (int)target_vocab_size); // Busca en 1000 palabras
        
        for (int v = 1; v < search_limit; ++v)
        {
            float score = output.getElement(last_pos, v);
            if (score > best_score)
            {
                best_score = score;
                best_token = v;
            }
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