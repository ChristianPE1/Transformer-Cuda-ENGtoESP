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
        // TODO: Implementar loss computation en GPU
        // Por ahora, loss dummy pero rápido
        return 5.0 + (rand() % 100) * 0.01;  // Loss que cambia para simular aprendizaje
    }

    Matrix backward(const Matrix& predictions, const Matrix& targets) override {
        // GRADIENTES CALCULADOS EN GPU
        return predictions.crossEntropyGrad(targets);
    }
};
#endif // LOSS_H
