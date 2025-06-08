#!/bin/bash

# Script to train the CUDA Transformer model

# Set the number of epochs and batch size
EPOCHS=10
BATCH_SIZE=32

# Set the paths for training and validation data
TRAIN_DATA="../data/train.txt"
VALID_DATA="../data/test.txt"

# Set the output directory for model checkpoints
OUTPUT_DIR="../output"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Compile the CUDA code
nvcc -o transformer_train ../src/main.cu -I../include -L/usr/local/cuda/lib64 -lcudart

# Run the training
./transformer_train --train_data $TRAIN_DATA --valid_data $VALID_DATA --epochs $EPOCHS --batch_size $BATCH_SIZE --output_dir $OUTPUT_DIR

echo "Training completed!"