// src/data/vocab.cu
#include "vocab.cuh"
#include <unordered_map>
#include <string>
#include <vector>

__device__ int getWordId(const std::unordered_map<std::string, int>& word_to_id, const std::string& word) {
    auto it = word_to_id.find(word);
    return (it != word_to_id.end()) ? it->second : word_to_id.at("<unk>");
}

__device__ std::string getWord(const std::unordered_map<int, std::string>& id_to_word, int id) {
    auto it = id_to_word.find(id);
    return (it != id_to_word.end()) ? it->second : "<unk>";
}

__global__ void initializeVocab(std::unordered_map<std::string, int>* word_to_id, 
                                 std::unordered_map<int, std::string>* id_to_word) {
    // Initialize special tokens
    (*word_to_id)["<pad>"] = 0;
    (*word_to_id)["<sos>"] = 1;
    (*word_to_id)["<eos>"] = 2;
    (*word_to_id)["<unk>"] = 3;

    (*id_to_word)[0] = "<pad>";
    (*id_to_word)[1] = "<sos>";
    (*id_to_word)[2] = "<eos>";
    (*id_to_word)[3] = "<unk>";
}

Vocab::Vocab() {
    // Initialize vocabularies on the host
    std::unordered_map<std::string, int> host_word_to_id;
    std::unordered_map<int, std::string> host_id_to_word;

    // Initialize vocabularies
    initializeVocab<<<1, 1>>>(&word_to_id, &id_to_word);
    cudaDeviceSynchronize();
}

void Vocab::addWord(const std::string& word) {
    if (word_to_id.find(word) == word_to_id.end()) {
        int id = word_to_id.size();
        word_to_id[word] = id;
        id_to_word[id] = word;
    }
}

int Vocab::getWordId(const std::string& word) {
    return getWordId(word_to_id, word);
}

std::string Vocab::getWord(int id) {
    return getWord(id_to_word, id);
}

size_t Vocab::size() const {
    return word_to_id.size();
}