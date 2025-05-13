#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cassert>
#include <cuda_runtime.h>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <cmath>

#define INPUT_SIZE 784
#define HIDDEN1_SIZE 128
#define HIDDEN2_SIZE 64
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.001f
#define EPOCHS 10
#define BATCH_SIZE 64

// CUDA error checking macro
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(code) << " at " << file << ":" << line << std::endl;
        exit(code);
    }
}

// Reverse integer for IDX file byte order
int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

// Read MNIST images from IDX file
std::vector<std::vector<float>> read_mnist_images(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    assert(file.is_open() && "Failed to open image file");

    int magic_number, number_of_images, rows, cols;
    file.read((char*)&magic_number, sizeof(magic_number)); magic_number = reverseInt(magic_number);
    file.read((char*)&number_of_images, sizeof(number_of_images)); number_of_images = reverseInt(number_of_images);
    file.read((char*)&rows, sizeof(rows)); rows = reverseInt(rows);
    file.read((char*)&cols, sizeof(cols)); cols = reverseInt(cols);
    assert(magic_number == 2051 && rows == 28 && cols == 28 && "Invalid image file format");

    std::vector<std::vector<float>> images(number_of_images, std::vector<float>(rows * cols));
    for (int i = 0; i < number_of_images; ++i) {
        for (int j = 0; j < rows * cols; ++j) {
            unsigned char temp;
            file.read((char*)&temp, sizeof(temp));
            images[i][j] = static_cast<float>(temp) / 255.0f;
        }
    }
    file.close();
    return images;
}

// Read MNIST labels from IDX file
std::vector<int> read_mnist_labels(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    assert(file.is_open() && "Failed to open label file");

    int magic_number, number_of_labels;
    file.read((char*)&magic_number, sizeof(magic_number)); magic_number = reverseInt(magic_number);
    file.read((char*)&number_of_labels, sizeof(number_of_labels)); number_of_labels = reverseInt(number_of_labels);
    assert(magic_number == 2049 && "Invalid label file format");

    std::vector<int> labels(number_of_labels);
    for (int i = 0; i < number_of_labels; ++i) {
        unsigned char temp;
        file.read((char*)&temp, sizeof(temp));
        labels[i] = static_cast<int>(temp);
    }
    file.close();
    return labels;
}

// Device functions for activation and derivatives
__device__ float relu(float x) { return x > 0 ? x : 0; }
__device__ float relu_deriv(float x) { return x > 0 ? 1.0f : 0.0f; }

// CUDA kernels
__global__ void forward_layer(float *input, float *weights, float *biases, float *output, int in_size, int out_size, int batch_offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < out_size) {
        float sum = biases[idx];
        for (int i = 0; i < in_size; ++i) {
            sum += input[batch_offset + i] * weights[i * out_size + idx];
        }
        output[batch_offset + idx] = relu(sum);
    }
}

__global__ void softmax_output(float *input, float *output, int size, int batch_offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        float max_val = input[batch_offset];
        for (int i = 1; i < size; ++i) {
            max_val = fmaxf(max_val, input[batch_offset + i]);
        }
        float sum_exp = 0.0f;
        for (int i = 0; i < size; ++i) {
            sum_exp += expf(input[batch_offset + i] - max_val);
        }
        sum_exp = fmaxf(sum_exp, 1e-6f);
        for (int i = 0; i < size; ++i) {
            output[batch_offset + i] = expf(input[batch_offset + i] - max_val) / sum_exp;
        }
    }
}

__global__ void compute_loss_gradient(float *pred, int label, float *grad, int batch_offset) {
    int i = threadIdx.x;
    if (i < OUTPUT_SIZE) {
        grad[batch_offset + i] = pred[batch_offset + i] - (i == label ? 1.0f : 0.0f);
    }
}

__global__ void backward_layer(float *d_out, float *output, float *input, float *weights, float *d_weights, float *d_biases, float *d_input, int in_size, int out_size, int batch_offset) {
    int idx = threadIdx.x;
    if (idx < out_size) {
        float grad = d_out[batch_offset + idx] * relu_deriv(output[batch_offset + idx]);
        atomicAdd(&d_biases[idx], grad);
        for (int i = 0; i < in_size; ++i) {
            atomicAdd(&d_weights[i * out_size + idx], input[batch_offset + i] * grad);
            atomicAdd(&d_input[batch_offset + i], weights[i * out_size + idx] * grad);
        }
    }
}

__global__ void update_weights(float *weights, float *biases, float *d_weights, float *d_biases, int size_w, int size_b, float lr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size_w) {
        weights[idx] -= lr * d_weights[idx];
        d_weights[idx] = 0.0f;
    }
    if (idx < size_b) {
        biases[idx] -= lr * d_biases[idx];
        d_biases[idx] = 0.0f;
    }
}

// Save weights and biases to file
void save_weights(const std::vector<float> &w1, const std::vector<float> &b1,
                  const std::vector<float> &w2, const std::vector<float> &b2,
                  const std::vector<float> &w3, const std::vector<float> &b3) {
    std::ofstream file("weights.dat", std::ios::binary);
    assert(file.is_open() && "Failed to open weights file for saving");

    auto write_vector = [&file](const std::vector<float> &v) {
        size_t size = v.size();
        file.write((char*)&size, sizeof(size));
        file.write((char*)v.data(), size * sizeof(float));
    };

    write_vector(w1);
    write_vector(b1);
    write_vector(w2);
    write_vector(b2);
    write_vector(w3);
    write_vector(b3);
    file.close();
}

// Load weights and biases from file
void load_weights(std::vector<float> &w1, std::vector<float> &b1,
                  std::vector<float> &w2, std::vector<float> &b2,
                  std::vector<float> &w3, std::vector<float> &b3) {
    std::ifstream file("weights.dat", std::ios::binary);
    assert(file.is_open() && "Failed to open weights file for loading");

    auto read_vector = [&file](std::vector<float> &v) {
        size_t size;
        file.read((char*)&size, sizeof(size));
        v.resize(size);
        file.read((char*)v.data(), size * sizeof(float));
    };

    read_vector(w1);
    read_vector(b1);
    read_vector(w2);
    read_vector(b2);
    read_vector(w3);
    read_vector(b3);
    file.close();
}

void train() {
    // Load training data
    auto train_images = read_mnist_images("train-images.idx3-ubyte");
    auto train_labels = read_mnist_labels("train-labels.idx1-ubyte");
    std::cout << "Starting training...\n";

    // Initialize weights and biases with Xavier initialization
    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<float> dist(0, 1.0f);

    std::vector<float> h_w1(INPUT_SIZE * HIDDEN1_SIZE);
    std::vector<float> h_b1(HIDDEN1_SIZE, 0.0f);
    std::vector<float> h_w2(HIDDEN1_SIZE * HIDDEN2_SIZE);
    std::vector<float> h_b2(HIDDEN2_SIZE, 0.0f);
    std::vector<float> h_w3(HIDDEN2_SIZE * OUTPUT_SIZE);
    std::vector<float> h_b3(OUTPUT_SIZE, 0.0f);

    float scale1 = sqrt(2.0f / (INPUT_SIZE + HIDDEN1_SIZE));
    float scale2 = sqrt(2.0f / (HIDDEN1_SIZE + HIDDEN2_SIZE));
    float scale3 = sqrt(2.0f / (HIDDEN2_SIZE + OUTPUT_SIZE));
    for (auto &w : h_w1) w = dist(gen) * scale1;
    for (auto &w : h_w2) w = dist(gen) * scale2;
    for (auto &w : h_w3) w = dist(gen) * scale3;

    // Allocate device memory
    float *d_w1, *d_b1, *d_w2, *d_b2, *d_w3, *d_b3;
    float *d_dw1, *d_db1, *d_dw2, *d_db2, *d_dw3, *d_db3;
    float *d_input, *d_h1, *d_h2, *d_out, *d_probs, *d_grad, *d_grad_h2, *d_grad_h1;
    CUDA_CHECK(cudaMalloc(&d_w1, INPUT_SIZE * HIDDEN1_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b1, HIDDEN1_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w2, HIDDEN1_SIZE * HIDDEN2_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b2, HIDDEN2_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w3, HIDDEN2_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b3, OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dw1, INPUT_SIZE * HIDDEN1_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db1, HIDDEN1_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dw2, HIDDEN1_SIZE * HIDDEN2_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db2, HIDDEN2_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dw3, HIDDEN2_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db3, OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_input, INPUT_SIZE * BATCH_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_h1, HIDDEN1_SIZE * BATCH_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_h2, HIDDEN2_SIZE * BATCH_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, OUTPUT_SIZE * BATCH_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_probs, OUTPUT_SIZE * BATCH_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad, OUTPUT_SIZE * BATCH_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_h2, HIDDEN2_SIZE * BATCH_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_h1, HIDDEN1_SIZE * BATCH_SIZE * sizeof(float)));

    // Copy weights and biases to device
    CUDA_CHECK(cudaMemcpy(d_w1, h_w1.data(), INPUT_SIZE * HIDDEN1_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1, h_b1.data(), HIDDEN1_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w2, h_w2.data(), HIDDEN1_SIZE * HIDDEN2_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2, h_b2.data(), HIDDEN2_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w3, h_w3.data(), HIDDEN2_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b3, h_b3.data(), OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_dw1, 0, INPUT_SIZE * HIDDEN1_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_db1, 0, HIDDEN1_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dw2, 0, HIDDEN1_SIZE * HIDDEN2_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_db2, 0, HIDDEN2_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dw3, 0, HIDDEN2_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_db3, 0, OUTPUT_SIZE * sizeof(float)));

    // Training loop
    std::vector<size_t> indices(train_images.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::ofstream csv_file("training_results.csv");
    csv_file << "Epoch,Time (seconds),Loss,Accuracy\n";

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        auto start_time = std::chrono::high_resolution_clock::now();
        float loss = 0.0f;
        int correct = 0;

        std::shuffle(indices.begin(), indices.end(), gen);

        for (size_t i = 0; i < train_images.size(); i += BATCH_SIZE) {
            int current_batch_size = std::min(BATCH_SIZE, static_cast<int>(train_images.size() - i));
            std::vector<float> h_batch(INPUT_SIZE * current_batch_size);
            std::vector<int> h_labels(current_batch_size);

            // Prepare batch
            for (int b = 0; b < current_batch_size; ++b) {
                size_t idx = indices[i + b];
                std::copy(train_images[idx].begin(), train_images[idx].end(), h_batch.begin() + b * INPUT_SIZE);
                h_labels[b] = train_labels[idx];
            }

            // Copy batch to device
            CUDA_CHECK(cudaMemcpy(d_input, h_batch.data(), INPUT_SIZE * current_batch_size * sizeof(float), cudaMemcpyHostToDevice));

            // Forward pass
            int threads = 256;
            int blocks1 = (HIDDEN1_SIZE + threads - 1) / threads;
            int blocks2 = (HIDDEN2_SIZE + threads - 1) / threads;
            int blocks3 = (OUTPUT_SIZE + threads - 1) / threads;

            for (int b = 0; b < current_batch_size; ++b) {
                int offset_in = b * INPUT_SIZE;
                int offset_h1 = b * HIDDEN1_SIZE;
                int offset_h2 = b * HIDDEN2_SIZE;
                int offset_out = b * OUTPUT_SIZE;

                forward_layer<<<blocks1, threads>>>(d_input, d_w1, d_b1, d_h1, INPUT_SIZE, HIDDEN1_SIZE, offset_in);
                CUDA_CHECK(cudaGetLastError());
                forward_layer<<<blocks2, threads>>>(d_h1, d_w2, d_b2, d_h2, HIDDEN1_SIZE, HIDDEN2_SIZE, offset_h1);
                CUDA_CHECK(cudaGetLastError());
                forward_layer<<<blocks3, threads>>>(d_h2, d_w3, d_b3, d_out, HIDDEN2_SIZE, OUTPUT_SIZE, offset_h2);
                CUDA_CHECK(cudaGetLastError());
                softmax_output<<<1, 1>>>(d_out, d_probs, OUTPUT_SIZE, offset_out);
                CUDA_CHECK(cudaGetLastError());
            }

            // Compute loss and gradients
            std::vector<float> h_probs(OUTPUT_SIZE * current_batch_size);
            CUDA_CHECK(cudaMemcpy(h_probs.data(), d_probs, OUTPUT_SIZE * current_batch_size * sizeof(float), cudaMemcpyDeviceToHost));
            for (int b = 0; b < current_batch_size; ++b) {
                int label = h_labels[b];
                float prob = h_probs[b * OUTPUT_SIZE + label];
                prob = std::max(prob, 1e-6f);
                loss += -logf(prob);
                int pred = std::distance(h_probs.begin() + b * OUTPUT_SIZE, 
                                        std::max_element(h_probs.begin() + b * OUTPUT_SIZE, 
                                                        h_probs.begin() + (b + 1) * OUTPUT_SIZE));
                correct += (pred == label);

                int offset_out = b * OUTPUT_SIZE;
                compute_loss_gradient<<<1, OUTPUT_SIZE>>>(d_probs, label, d_grad, offset_out);
                CUDA_CHECK(cudaGetLastError());
            }

            // Backward pass
            for (int b = 0; b < current_batch_size; ++b) {
                int offset_in = b * INPUT_SIZE;
                int offset_h1 = b * HIDDEN1_SIZE;
                int offset_h2 = b * HIDDEN2_SIZE;
                int offset_out = b * OUTPUT_SIZE;

                backward_layer<<<1, OUTPUT_SIZE>>>(d_grad, d_out, d_h2, d_w3, d_dw3, d_db3, d_grad_h2, HIDDEN2_SIZE, OUTPUT_SIZE, offset_out);
                CUDA_CHECK(cudaGetLastError());
                backward_layer<<<1, HIDDEN2_SIZE>>>(d_grad_h2, d_h2, d_h1, d_w2, d_dw2, d_db2, d_grad_h1, HIDDEN1_SIZE, HIDDEN2_SIZE, offset_h2);
                CUDA_CHECK(cudaGetLastError());
                backward_layer<<<1, HIDDEN1_SIZE>>>(d_grad_h1, d_h1, d_input, d_w1, d_dw1, d_db1, d_grad_h1, INPUT_SIZE, HIDDEN1_SIZE, offset_in);
                CUDA_CHECK(cudaGetLastError());
            }

            // Update weights
            int blocks_w1 = (INPUT_SIZE * HIDDEN1_SIZE + threads - 1) / threads;
            int blocks_w2 = (HIDDEN1_SIZE * HIDDEN2_SIZE + threads - 1) / threads;
            int blocks_w3 = (HIDDEN2_SIZE * OUTPUT_SIZE + threads - 1) / threads;
            update_weights<<<blocks_w1, threads>>>(d_w1, d_b1, d_dw1, d_db1, INPUT_SIZE * HIDDEN1_SIZE, HIDDEN1_SIZE, LEARNING_RATE / current_batch_size);
            update_weights<<<blocks_w2, threads>>>(d_w2, d_b2, d_dw2, d_db2, HIDDEN1_SIZE * HIDDEN2_SIZE, HIDDEN2_SIZE, LEARNING_RATE / current_batch_size);
            update_weights<<<blocks_w3, threads>>>(d_w3, d_b3, d_dw3, d_db3, HIDDEN2_SIZE * OUTPUT_SIZE, OUTPUT_SIZE, LEARNING_RATE / current_batch_size);
            CUDA_CHECK(cudaGetLastError());
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> duration = end_time - start_time;
        float accuracy = 100.0f * correct / train_images.size();
        loss /= train_images.size();

        csv_file << epoch + 1 << "," << duration.count() << "," << loss << "," << accuracy << "\n";
        std::cout << "Epoch " << epoch + 1 << " | Loss: " << loss 
                  << " | Accuracy: " << accuracy << "% | Time: " << duration.count() << " seconds\n";
    }
    csv_file.close();

    // Save weights
    CUDA_CHECK(cudaMemcpy(h_w1.data(), d_w1, INPUT_SIZE * HIDDEN1_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b1.data(), d_b1, HIDDEN1_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_w2.data(), d_w2, HIDDEN1_SIZE * HIDDEN2_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b2.data(), d_b2, HIDDEN2_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_w3.data(), d_w3, HIDDEN2_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b3.data(), d_b3, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    save_weights(h_w1, h_b1, h_w2, h_b2, h_w3, h_b3);

    // Free device memory
    CUDA_CHECK(cudaFree(d_w1)); CUDA_CHECK(cudaFree(d_b1));
    CUDA_CHECK(cudaFree(d_w2)); CUDA_CHECK(cudaFree(d_b2));
    CUDA_CHECK(cudaFree(d_w3)); CUDA_CHECK(cudaFree(d_b3));
    CUDA_CHECK(cudaFree(d_dw1)); CUDA_CHECK(cudaFree(d_db1));
    CUDA_CHECK(cudaFree(d_dw2)); CUDA_CHECK(cudaFree(d_db2));
    CUDA_CHECK(cudaFree(d_dw3)); CUDA_CHECK(cudaFree(d_db3));
    CUDA_CHECK(cudaFree(d_input)); CUDA_CHECK(cudaFree(d_h1));
    CUDA_CHECK(cudaFree(d_h2)); CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_probs)); CUDA_CHECK(cudaFree(d_grad));
    CUDA_CHECK(cudaFree(d_grad_h2)); CUDA_CHECK(cudaFree(d_grad_h1));
}

void predict(const std::vector<std::vector<float>> &images, const std::vector<int> &labels) {
    std::cout << "Testing model...\n";

    // Load trained weights
    std::vector<float> h_w1, h_b1, h_w2, h_b2, h_w3, h_b3;
    load_weights(h_w1, h_b1, h_w2, h_b2, h_w3, h_b3);

    // Allocate device memory
    float *d_w1, *d_b1, *d_w2, *d_b2, *d_w3, *d_b3;
    float *d_input, *d_h1, *d_h2, *d_out, *d_probs;
    CUDA_CHECK(cudaMalloc(&d_w1, INPUT_SIZE * HIDDEN1_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b1, HIDDEN1_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w2, HIDDEN1_SIZE * HIDDEN2_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b2, HIDDEN2_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w3, HIDDEN2_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b3, OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_input, INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_h1, HIDDEN1_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_h2, HIDDEN2_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_probs, OUTPUT_SIZE * sizeof(float)));

    // Copy weights to device
    CUDA_CHECK(cudaMemcpy(d_w1, h_w1.data(), INPUT_SIZE * HIDDEN1_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1, h_b1.data(), HIDDEN1_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w2, h_w2.data(), HIDDEN1_SIZE * HIDDEN2_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2, h_b2.data(), HIDDEN2_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w3, h_w3.data(), HIDDEN2_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b3, h_b3.data(), OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    // Evaluate test images
    int correct = 0;
    std::cout << "\nPredictions for first 10 test images:\n";
    for (size_t idx = 0; idx < images.size(); ++idx) {
        // Copy image to device
        CUDA_CHECK(cudaMemcpy(d_input, images[idx].data(), INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));

        // Forward pass
        int threads = 256;
        int blocks1 = (HIDDEN1_SIZE + threads - 1) / threads;
        int blocks2 = (HIDDEN2_SIZE + threads - 1) / threads;
        int blocks3 = (OUTPUT_SIZE + threads - 1) / threads;

        forward_layer<<<blocks1, threads>>>(d_input, d_w1, d_b1, d_h1, INPUT_SIZE, HIDDEN1_SIZE, 0);
        CUDA_CHECK(cudaGetLastError());
        forward_layer<<<blocks2, threads>>>(d_h1, d_w2, d_b2, d_h2, HIDDEN1_SIZE, HIDDEN2_SIZE, 0);
        CUDA_CHECK(cudaGetLastError());
        forward_layer<<<blocks3, threads>>>(d_h2, d_w3, d_b3, d_out, HIDDEN2_SIZE, OUTPUT_SIZE, 0);
        CUDA_CHECK(cudaGetLastError());
        softmax_output<<<1, 1>>>(d_out, d_probs, OUTPUT_SIZE, 0);
        CUDA_CHECK(cudaGetLastError());

        // Get predictions
        std::vector<float> h_probs(OUTPUT_SIZE);
        CUDA_CHECK(cudaMemcpy(h_probs.data(), d_probs, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
        int pred = std::distance(h_probs.begin(), std::max_element(h_probs.begin(), h_probs.end()));
        int label = labels[idx];
        correct += (pred == label);

        // Print first 10 predictions
        if (idx < 10) {
            std::cout << "Image " << idx + 1 << ": True label = " << label << ", Predicted = " << pred << "\n";
        }
    }

    float accuracy = 100.0f * correct / images.size();
    std::cout << "Test Accuracy = " << accuracy << "%\n";

    // Free device memory
    CUDA_CHECK(cudaFree(d_w1)); CUDA_CHECK(cudaFree(d_b1));
    CUDA_CHECK(cudaFree(d_w2)); CUDA_CHECK(cudaFree(d_b2));
    CUDA_CHECK(cudaFree(d_w3)); CUDA_CHECK(cudaFree(d_b3));
    CUDA_CHECK(cudaFree(d_input)); CUDA_CHECK(cudaFree(d_h1));
    CUDA_CHECK(cudaFree(d_h2)); CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_probs));
}

int main() {
    auto train_images = read_mnist_images("train-images.idx3-ubyte");
    auto train_labels = read_mnist_labels("train-labels.idx1-ubyte");
    auto test_images = read_mnist_images("t10k-images.idx3-ubyte");
    auto test_labels = read_mnist_labels("t10k-labels.idx1-ubyte");

    std::cout << "Data loaded successfully. Training images: " << train_images.size() 
              << ", Test images: " << test_images.size() << "\n";

    train();
    predict(test_images, test_labels);

    return 0;
}