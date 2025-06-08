// filepath: cuda-transformer/cuda-transformer/src/training/trainer.cu
#include "trainer.cuh"
#include "../transformer/transformer.cuh"
#include "../data/dataset.cuh"
#include "../training/loss.cuh"
#include "../training/optimizer.cuh"
#include <iostream>

__global__ void trainKernel(Transformer transformer, Dataset dataset, Optimizer optimizer, Loss lossFunction, int num_epochs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dataset.size()) {
        // Load input and target sequences
        auto input = dataset.getInput(idx);
        auto target = dataset.getTarget(idx);

        // Forward pass
        Matrix output = transformer.forward(input, target);

        // Compute loss
        float loss = lossFunction.calculate(output, target);

        // Backward pass and optimization
        optimizer.step(transformer, loss);
    }
}

void Trainer::train(Transformer &transformer, Dataset &dataset, Optimizer &optimizer, Loss &lossFunction, int num_epochs) {
    int num_samples = dataset.size();
    int blockSize = 256;
    int numBlocks = (num_samples + blockSize - 1) / blockSize;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        std::cout << "Epoch " << epoch + 1 << "/" << num_epochs << std::endl;

        // Launch the training kernel
        trainKernel<<<numBlocks, blockSize>>>(transformer, dataset, optimizer, lossFunction, num_epochs);
        cudaDeviceSynchronize();

        // Check for errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
            break;
        }
    }
}