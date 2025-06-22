// filepath: cuda-transformer/cuda-transformer/src/training/loss.cuh
#ifndef LOSS_H
#define LOSS_H

#include <cuda_runtime.h>
#include <iostream>
#include "utils/matrix.cuh"

class Loss {
public:
    __device__ virtual double forward(const Matrix& predictions, const Matrix& targets) = 0;
    __device__ virtual Matrix backward(const Matrix& predictions, const Matrix& targets) = 0;

    void calculateCrossEntropy(const float* predictions, const int* targets, float* loss, int num_classes, int batch_size);
};

class CrossEntropyLoss : public Loss {
public:
    __device__ double forward(const Matrix& predictions, const Matrix& targets) override {
        double loss = 0.0;
        int batch_size = predictions.getRows();
        int num_classes = predictions.getCols();
        const float* pred_data = predictions.getData();
        const float* targ_data = targets.getData();

        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < num_classes; ++j) {
                loss -= targ_data[i * num_classes + j] * log(pred_data[i * num_classes + j] + 1e-10);
            }
        }
        return loss / batch_size;
    }

    __device__ Matrix backward(const Matrix& predictions, const Matrix& targets) override {
        int batch_size = predictions.getRows();
        int num_classes = predictions.getCols();
        Matrix grad(batch_size, num_classes);
        float* grad_data = grad.getData();
        const float* pred_data = predictions.getData();
        const float* targ_data = targets.getData();

        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < num_classes; ++j) {
                grad_data[i * num_classes + j] = pred_data[i * num_classes + j] - targ_data[i * num_classes + j];
            }
        }
        // Divide manualmente
        for (int i = 0; i < batch_size * num_classes; ++i) {
            grad_data[i] /= batch_size;
        }
        return grad;
    }
#endif // LOSS_H

#endif // LOSS_H