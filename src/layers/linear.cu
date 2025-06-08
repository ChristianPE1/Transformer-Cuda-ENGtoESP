// filepath: cuda-transformer/cuda-transformer/src/layers/linear.cu
#include "linear.cuh"
#include "cuda_utils.cuh"

__global__ void linear_forward_kernel(const float *input, const float *weights, const float *bias, float *output, int input_dim, int output_dim, int batch_size) {
    int batch_index = blockIdx.x;
    int output_index = threadIdx.x;

    if (batch_index < batch_size && output_index < output_dim) {
        float sum = 0.0f;
        for (int i = 0; i < input_dim; ++i) {
            sum += input[batch_index * input_dim + i] * weights[i * output_dim + output_index];
        }
        output[batch_index * output_dim + output_index] = sum + bias[output_index];
    }
}

void Linear::forward(const Matrix &input, const Matrix &weights, const Matrix &bias, Matrix &output) {
    int batch_size = input.getRows();
    int input_dim = input.getCols();
    int output_dim = weights.getCols();

    // Allocate device memory
    float *d_input, *d_weights, *d_bias, *d_output;
    cudaMalloc(&d_input, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_weights, input_dim * output_dim * sizeof(float));
    cudaMalloc(&d_bias, output_dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * output_dim * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_input, input.getData(), batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights.getData(), input_dim * output_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias.getData(), output_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    linear_forward_kernel<<<batch_size, output_dim>>>(d_input, d_weights, d_bias, d_output, input_dim, output_dim, batch_size);

    // Copy output back to host
    cudaMemcpy(output.getData(), d_output, batch_size * output_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_bias);
    cudaFree(d_output);
}