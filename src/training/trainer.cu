// filepath: cuda-transformer/cuda-transformer/src/training/trainer.cu
#include "trainer.cuh"
#include <iostream>

void Trainer::train(const std::vector<std::vector<int>>& source_batches, const std::vector<std::vector<int>>& target_batches) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::cout << "Epoch " << (epoch + 1) << "/" << epochs << std::endl;
        for (size_t i = 0; i < source_batches.size(); ++i) {
            // 1. Forward
            Matrix output = model.forward(source_batches[i], target_batches[i]);
            // 2. Compute loss
            double loss = loss_fn.forward(output, /*targets as Matrix*/); // Debes convertir target_batches[i] a Matrix
            // 3. Backward y optimizer
            Matrix grad = loss_fn.backward(output, /*targets as Matrix*/);
            // optimizer.step(...); // Debes pasar los parámetros y gradientes correctos
        }
    }
}