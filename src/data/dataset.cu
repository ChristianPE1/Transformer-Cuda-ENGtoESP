// src/data/dataset.cu
#include "dataset.cuh"
#include <fstream>
#include <iostream>
#include <vector>

Dataset::Dataset(const std::string &file_path) {
    loadDataset(file_path);
}

void Dataset::loadDataset(const std::string &file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << file_path << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        // Simple tokenization by whitespace
        std::vector<std::string> tokens;
        std::istringstream iss(line);
        std::string token;
        while (iss >> token) {
            tokens.push_back(token);
        }
        data.push_back(tokens);
    }

    file.close();
}

size_t Dataset::size() const {
    return data.size();
}

const std::vector<std::string>& Dataset::getExample(size_t index) const {
    return data.at(index);
}