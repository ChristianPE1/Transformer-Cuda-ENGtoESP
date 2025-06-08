#include <iostream>
#include <cuda_runtime.h>
#include "data/dataset.cuh"
#include "data/vocab.cuh"

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
        auto eng_vocab = dataset.getEngVocab();
        auto spa_vocab = dataset.getSpaVocab();

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
            auto [eng_ids, spa_ids] = batch[i];
            std::cout << "Sample " << i << ":" << std::endl;
            std::cout << "  ENG: " << eng_vocab.idsToSentence(eng_ids) << std::endl;
            std::cout << "  SPA: " << spa_vocab.idsToSentence(spa_ids) << std::endl;
        }

        std::cout << "\n=== Success! Dataset ready for training ===" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}