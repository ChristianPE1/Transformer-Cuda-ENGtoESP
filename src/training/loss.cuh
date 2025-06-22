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
};

class CrossEntropyLoss : public Loss {
public:
    __device__ double forward(const Matrix& predictions, const Matrix& targets) override {
        double loss = 0.0;
        int batch_size = predictions.getRows();
        int num_classes = predictions.getCols();

        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < num_classes; ++j) {
                loss -= targets[i][j] * log(predictions[i][j] + 1e-10); // Adding epsilon to prevent log(0)
            }
        }
        return loss / batch_size;
    }

    __device__ Matrix backward(const Matrix& predictions, const Matrix& targets) override {
        int batch_size = predictions.getRows();
        int num_classes = predictions.getCols();
        Matrix grad(batch_size, num_classes);

        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < num_classes; ++j) {
                grad[i][j] = predictions[i][j] - targets[i][j];
            }
        }
        return grad / batch_size;
    }
};

#endif // LOSS_H