// filepath: cuda-transformer/cuda-transformer/src/training/trainer.cu
#include "trainer.cuh"
#include <iostream>

// Convierte un vector de índices a una matriz one-hot (batch_size x num_classes)
Matrix vectorToOneHotMatrix(const std::vector<int>& indices, int num_classes) {
    int batch_size = indices.size();
    Matrix mat(batch_size, num_classes, 0.0f);
    for (int i = 0; i < batch_size; ++i) {
        if (indices[i] >= 0 && indices[i] < num_classes)
            mat.setElement(i, indices[i], 1.0f);
    }
    return mat;
}

void Trainer::train(const std::vector<std::vector<int>>& source_batches, const std::vector<std::vector<int>>& target_batches) {
    int num_classes = model.getTargetVocabSize(); 
    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::cout << "Epoch " << (epoch + 1) << "/" << epochs << std::endl;
        for (size_t i = 0; i < source_batches.size(); ++i) {
            // 1. Forward
            Matrix output = model.forward(source_batches[i], target_batches[i]);
            Matrix target = vectorToOneHotMatrix(target_batches[i], num_classes);
            // 2. Compute loss
            double loss = loss_fn.forward(output, target);
            // 3. Backward y optimizer
            Matrix grad = loss_fn.backward(output, target);
            // optimizer.step(...); // Debes pasar los parámetros y gradientes correctos
        }
    }
}

Trainer::Trainer(Transformer& model, Optimizer& optimizer, Loss& loss_fn, int batch_size, int epochs)
    : model(model), optimizer(optimizer), loss_fn(loss_fn), batch_size(batch_size), epochs(epochs) {
    // Constructor implementation
}