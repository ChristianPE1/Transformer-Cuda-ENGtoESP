#!/bin/bash

# Script to evaluate the Transformer model on the test dataset

# Set the CUDA device (optional)
export CUDA_VISIBLE_DEVICES=0

# Define paths
DATA_DIR="../data"
MODEL_DIR="../models"
OUTPUT_DIR="../output"
TEST_FILE="$DATA_DIR/test.txt"
OUTPUT_FILE="$OUTPUT_DIR/evaluation_results.txt"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Run the evaluation
nvcc ../src/main.cu -o evaluate_transformer -I../include -L../lib -lcudart
./evaluate_transformer --test_file $TEST_FILE --model_dir $MODEL_DIR --output_file $OUTPUT_FILE

echo "Evaluation completed. Results saved to $OUTPUT_FILE."