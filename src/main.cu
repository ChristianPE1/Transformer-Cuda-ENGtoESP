// filepath: cuda-transformer/cuda-transformer/src/main.cu
#include <iostream>
#include "transformer/transformer.cuh"
#include "data/dataset.cuh"
#include "training/trainer.cuh"

int main() {
    try {
        std::cout << "=== Transformer CUDA Training ===" << std::endl;

        // Initialize dataset
        Dataset dataset("data/train.txt", "data/test.txt");
        dataset.load();

        // Create vocabularies
        Vocab eng_vocab, spa_vocab;
        dataset.buildVocab(eng_vocab, spa_vocab);

        std::cout << "Vocabulario ingles: " << eng_vocab.size() << " palabras" << std::endl;
        std::cout << "Vocabulario espanol: " << spa_vocab.size() << " palabras" << std::endl;

        // Create Transformer
        Transformer transformer(eng_vocab.size(), spa_vocab.size(), 128, 4, 2, 256);
        std::cout << "Transformer creado exitosamente!" << std::endl;

        // Initialize Trainer
        Trainer trainer(&transformer, &dataset);
        trainer.train(10); // Train for 10 epochs

        std::cout << "Entrenamiento completado!" << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}