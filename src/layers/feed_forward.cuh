// filepath: cuda-transformer/cuda-transformer/src/layers/feed_forward.cuh
#ifndef FEED_FORWARD_H
#define FEED_FORWARD_H

#include "../utils/matrix.cuh"

class FeedForward {
private:
    size_t d_model;
    size_t d_ff;
    Matrix W1, W2;  // Weight matrices

public:
    FeedForward(size_t d_model, size_t d_ff = 2048);
    ~FeedForward();

    Matrix forward(const Matrix &input);
    void initializeWeights();
    
    // For gradient updates
    void updateWeights(float learning_rate);
    
    size_t getDModel() const { return d_model; }
    size_t getDFF() const { return d_ff; }
};

#endif // FEED_FORWARD_H