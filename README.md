# CUDA Transformer

This project implements a Transformer model using CUDA for efficient training and inference. The Transformer architecture is widely used for natural language processing tasks, particularly for translation.

## Project Structure

The project is organized into several directories and files:

- **src/**: Contains the source code for the CUDA implementation.
  - **main.cu**: Entry point of the application, initializes the CUDA environment and orchestrates training and evaluation.
  - **transformer/**: Contains the implementation of the Transformer model.
    - **transformer.cu**: Implementation of the Transformer class.
    - **transformer.cuh**: Header file for the Transformer class.
    - **attention.cu**: Implementation of the multi-head attention mechanism.
    - **attention.cuh**: Header file for the MultiHeadAttention class.
    - **embeddings.cu**: Implementation of the embedding layer.
    - **embeddings.cuh**: Header file for the Embedding class.
    - **encoder.cu**: Implementation of the encoder layers.
    - **encoder.cuh**: Header file for the EncoderLayer class.
    - **decoder.cu**: Implementation of the decoder layers.
    - **decoder.cuh**: Header file for the DecoderLayer class.
  - **layers/**: Contains implementations of various layers used in the Transformer.
    - **layer_norm.cu**: Implementation of layer normalization.
    - **layer_norm.cuh**: Header file for the LayerNorm class.
    - **feed_forward.cu**: Implementation of the feed-forward network.
    - **feed_forward.cuh**: Header file for the FeedForward class.
    - **linear.cu**: Implementation of the linear layer.
    - **linear.cuh**: Header file for the Linear class.
  - **utils/**: Contains utility functions and classes.
    - **matrix.cu**: Implementation of matrix operations.
    - **matrix.cuh**: Header file for the Matrix class.
    - **cuda_utils.cu**: Utility functions for managing CUDA memory.
    - **cuda_utils.cuh**: Header file for CUDA utility functions.
    - **mask_utils.cu**: Functions for creating masks.
    - **mask_utils.cuh**: Header file for the MaskUtils class.
  - **training/**: Contains the training loop and related components.
    - **trainer.cu**: Implementation of the training loop.
    - **trainer.cuh**: Header file for the Trainer class.
    - **loss.cu**: Implementation of loss functions.
    - **loss.cuh**: Header file for the Loss class.
    - **optimizer.cu**: Implementation of optimization algorithms.
    - **optimizer.cuh**: Header file for the Optimizer class.
  - **data/**: Contains classes for managing datasets and vocabularies.
    - **vocab.cu**: Implementation of vocabulary management.
    - **vocab.cuh**: Header file for the Vocab class.
    - **dataset.cu**: Implementation of dataset loading and preprocessing.
    - **dataset.cuh**: Header file for the Dataset class.

- **include/**: Contains common definitions and includes for the project.
  - **common.cuh**: Common definitions used across the project.

- **data/**: Contains the training and test data files.
  - **train.txt**: Training data for the translation task.
  - **test.txt**: Test data for the translation task.

- **scripts/**: Contains scripts for automating training and evaluation.
  - **train.sh**: Script to automate the training process.
  - **evaluate.sh**: Script to automate the evaluation process.

- **CMakeLists.txt**: Configuration file for CMake to build the project.

- **Makefile**: Build rules and targets for the project.

## Setup Instructions

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd cuda-transformer
   ```

2. **Install dependencies**:
   Ensure you have CUDA installed on your machine. Follow the installation instructions for your operating system.

3. **Build the project**:
   You can build the project using CMake or Makefile. For CMake:
   ```
   mkdir build
   cd build
   cmake ..
   make
   ```

4. **Run the training script**:
   ```
   ./scripts/train.sh
   ```

5. **Evaluate the model**:
   ```
   ./scripts/evaluate.sh
   ```

## Usage

After training, you can use the trained model for inference on new data. Modify the `main.cu` file to load the model and perform predictions as needed.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.