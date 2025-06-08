// dataset.cuh
#ifndef DATASET_CUH
#define DATASET_CUH

#include <vector>
#include <string>
#include "vocab.cuh"

class Dataset {
public:
    Dataset(const std::string& file_path, Vocab& vocab);
    void load();
    std::vector<std::vector<int>> getSourceSequences() const;
    std::vector<std::vector<int>> getTargetSequences() const;

private:
    std::string file_path;
    Vocab& vocab;
    std::vector<std::vector<int>> source_sequences;
    std::vector<std::vector<int>> target_sequences;

    void preprocessLine(const std::string& line, std::vector<int>& source, std::vector<int>& target);
};

#endif // DATASET_CUH