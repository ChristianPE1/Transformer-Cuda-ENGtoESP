// vocab.cuh
#ifndef VOCAB_H
#define VOCAB_H

#include <unordered_map>
#include <string>
#include <vector>

class Vocab {
public:
    Vocab();
    void addWord(const std::string &word);
    int getWordId(const std::string &word) const;
    std::string getWord(int id) const;
    size_t size() const;

private:
    std::unordered_map<std::string, int> word_to_id;
    std::unordered_map<int, std::string> id_to_word;
};

#endif // VOCAB_H