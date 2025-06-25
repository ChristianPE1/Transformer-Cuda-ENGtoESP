// filepath: cuda-transformer/cuda-transformer/src/training/loss.cuh
#ifndef LOSS_H
#define LOSS_H

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include "utils/matrix.cuh"

class Loss {
public:
    virtual double forward(const Matrix& predictions, const Matrix& targets) = 0;
    virtual Matrix backward(const Matrix& predictions, const Matrix& targets) = 0;

    void calculateCrossEntropy(const float* predictions, const int* targets, float* loss, int num_classes, int batch_size);
};

class CrossEntropyLoss : public Loss {
public:
    double forward(const Matrix& predictions, const Matrix& targets) override {
        int batch_size = predictions.getRows();
        int num_classes = predictions.getCols();
        
        double total_loss = 0.0;
        
        for (int i = 0; i < batch_size; ++i) {
            int target_class = (int)targets.getElement(i, 0);
            if (target_class >= 0 && target_class < num_classes) {
                
                // Aplicar softmax a la fila i de predictions
                float max_val = predictions.getElement(i, 0);
                for (int j = 1; j < num_classes; ++j) {
                    max_val = std::max(max_val, predictions.getElement(i, j));
                }
                
                float sum_exp = 0.0f;
                for (int j = 0; j < num_classes; ++j) {
                    sum_exp += exp(predictions.getElement(i, j) - max_val);
                }
                
                float target_prob = exp(predictions.getElement(i, target_class) - max_val) / sum_exp;
                total_loss -= log(target_prob + 1e-10f); // Evitar log(0)
            }
        }
        
        double avg_loss = total_loss / batch_size;
        
        std::cout << " [REAL-LOSS] batch:" << batch_size 
                  << " classes:" << num_classes 
                  << " loss:" << std::fixed << std::setprecision(2) << avg_loss;
                  
        return avg_loss;
    }

    Matrix backward(const Matrix& predictions, const Matrix& targets) override {
        int batch_size = predictions.getRows();
        int num_classes = predictions.getCols();
        
        Matrix grad(batch_size, num_classes, 0.0f);
        
        for (int i = 0; i < batch_size; ++i) {
            int target_class = (int)targets.getElement(i, 0);
            if (target_class >= 0 && target_class < num_classes) {
                
                // Calcular softmax para la fila i
                float max_val = predictions.getElement(i, 0);
                for (int j = 1; j < num_classes; ++j) {
                    max_val = std::max(max_val, predictions.getElement(i, j));
                }
                
                float sum_exp = 0.0f;
                std::vector<float> softmax_probs(num_classes);
                for (int j = 0; j < num_classes; ++j) {
                    softmax_probs[j] = exp(predictions.getElement(i, j) - max_val);
                    sum_exp += softmax_probs[j];
                }
                
                // Normalizar softmax
                for (int j = 0; j < num_classes; ++j) {
                    softmax_probs[j] /= sum_exp;
                }
                  // Gradiente de cross-entropy: softmax_prob - target (one-hot)
                for (int j = 0; j < num_classes; ++j) {
                    float gradient = softmax_probs[j];
                    if (j == target_class) {
                        gradient -= 1.0f; // Restar 1 para la clase correcta
                    }
                    // AMPLIFICAR los gradientes para que tengan más impacto
                    gradient *= 10.0f; // Multiplicar por 10 para mayor magnitud
                    grad.setElement(i, j, gradient / batch_size);
                }
            }
        }
        
        return grad;
    }
};
#endif // LOSS_H
