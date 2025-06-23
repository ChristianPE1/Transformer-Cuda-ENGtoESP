// filepath: cuda-transformer/cuda-transformer/src/training/loss.cuh
#ifndef LOSS_H
#define LOSS_H

#include <cuda_runtime.h>
#include <iostream>
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
        
        // LOSS REAL: Cross-entropy simplificado
        double total_loss = 0.0;
        
        for (int i = 0; i < batch_size; ++i) {
            // Target class para esta posición
            int target_class = (int)targets.getElement(i, 0);
            if (target_class >= 0 && target_class < num_classes) {
                // Probabilidad predicha para la clase correcta
                float predicted_prob = predictions.getElement(i, target_class);
                
                // Cross-entropy: -log(predicted_prob)
                total_loss += -log(predicted_prob + 1e-8f); // Epsilon para evitar log(0)
            }
        }
        
        double avg_loss = total_loss / batch_size;
        return avg_loss;
    }

    Matrix backward(const Matrix& predictions, const Matrix& targets) override {
        int batch_size = predictions.getRows();
        int num_classes = predictions.getCols();
        
        // GRADIENTES REALES
        Matrix grad(batch_size, num_classes, 0.0f);
        
        for (int i = 0; i < batch_size; ++i) {
            int target_class = (int)targets.getElement(i, 0);
            if (target_class >= 0 && target_class < num_classes) {
                // Gradiente = predicted_prob - 1 para la clase correcta
                float predicted_prob = predictions.getElement(i, target_class);
                grad.setElement(i, target_class, predicted_prob - 1.0f);
                
                // Gradiente = predicted_prob para otras clases
                for (int j = 0; j < num_classes; ++j) {
                    if (j != target_class) {
                        float prob = predictions.getElement(i, j);
                        grad.setElement(i, j, prob);
                    }
                }
            }
        }
        
        return grad;
    }
};
#endif // LOSS_H
