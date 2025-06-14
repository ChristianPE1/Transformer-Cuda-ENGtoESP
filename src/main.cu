#include <iostream>
#include <cuda_runtime.h>
#include "data/dataset.cuh"
#include "data/vocab.cuh"
#include "transformer/transformer.cuh"
#include "utils/matrix.cuh"

int main()
{
    try
    {
        std::cout << "=== CUDA Transformer with TSV Dataset ===" << std::endl;

        // Verify CUDA
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        std::cout << "CUDA devices found: " << deviceCount << std::endl;

        if (deviceCount == 0)
        {
            std::cerr << "No CUDA devices found!" << std::endl;
            return 1;
        }

        // Load and process dataset
        Dataset dataset;
        std::cout << "Loading TSV file..." << std::endl;
        dataset.loadTSV("db_translate.tsv");

        std::cout << "Building vocabularies..." << std::endl;
        dataset.buildVocabularies();

        std::cout << "Creating train/test split..." << std::endl;
        dataset.createTrainTestSplit(0.8f);

        std::cout << "\nDataset Statistics:" << std::endl;
        std::cout << "English vocab size: " << dataset.getEngVocab().size() << std::endl;
        std::cout << "Spanish vocab size: " << dataset.getSpaVocab().size() << std::endl;
        std::cout << "Training samples: " << dataset.getTrainSize() << std::endl;
        std::cout << "Test samples: " << dataset.getTestSize() << std::endl;

        // Test vocabulary
        std::cout << "\n=== Vocabulary Test ===" << std::endl;
        const auto &eng_vocab = dataset.getEngVocab();
        const auto &spa_vocab = dataset.getSpaVocab();

        // Test English sentence
        std::string test_eng = "<sos> hello world <eos>";
        auto eng_ids = eng_vocab.sentenceToIds(test_eng);
        std::cout << "English: \"" << test_eng << "\" -> ";
        for (int id : eng_ids)
        {
            std::cout << id << " ";
        }
        std::cout << "-> \"" << eng_vocab.idsToSentence(eng_ids) << "\"" << std::endl;

        // Test Spanish sentence
        std::string test_spa = "<sos> hola mundo <eos>";
        auto spa_ids = spa_vocab.sentenceToIds(test_spa);
        std::cout << "Spanish: \"" << test_spa << "\" -> ";
        for (int id : spa_ids)
        {
            std::cout << id << " ";
        }
        std::cout << "-> \"" << spa_vocab.idsToSentence(spa_ids) << "\"" << std::endl;

        // Test batch loading
        std::cout << "\n=== Batch Test ===" << std::endl;
        auto batch = dataset.getBatch(3, true);
        std::cout << "Loaded batch of " << batch.size() << " samples:" << std::endl;

        for (size_t i = 0; i < batch.size(); ++i)
        {
            const auto &sample = batch[i];
            const auto &eng_ids = sample.first;
            const auto &spa_ids = sample.second;
            std::cout << "Sample " << i << ":" << std::endl;
            std::cout << "  ENG: " << eng_vocab.idsToSentence(eng_ids) << std::endl;
            std::cout << "  SPA: " << spa_vocab.idsToSentence(spa_ids) << std::endl;
        }

        std::cout << "\n=== Success! Dataset ready for training ===" << std::endl;

        // Test Transformer
        std::cout << "\n=== Testing Transformer ===" << std::endl;
        Transformer transformer(dataset.getEngVocab().size(),
                                dataset.getSpaVocab().size(),
                                128,  // d_model
                                4,    // n_heads
                                2,    // n_layers
                                256); // d_ff

        // Test forward pass
        auto test_batch = dataset.getBatch(1, true);
        if (!test_batch.empty())
        {
            const auto &sample = test_batch[0];
            const auto &source_ids = sample.first;
            const auto &target_ids = sample.second;

            std::cout << "Testing forward pass with:" << std::endl;
            std::cout << "  Source: " << eng_vocab.idsToSentence(source_ids) << std::endl;
            std::cout << "  Target: " << spa_vocab.idsToSentence(target_ids) << std::endl;

            Matrix output = transformer.forward(source_ids, target_ids);
            std::cout << "Forward pass completed!" << std::endl;
            std::cout << "Output shape: " << output.getRows() << "x" << output.getCols() << std::endl;

            // Test generation
            std::cout << "\nTesting generation..." << std::endl;
            auto generated = transformer.generate(source_ids, 2, 3, 10); // sos=2, eos=3
            std::cout << "Generated: " << spa_vocab.idsToSentence(generated) << std::endl;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}