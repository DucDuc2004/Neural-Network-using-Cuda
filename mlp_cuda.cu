#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <chrono>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#define CUDA_CHECK(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                  << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
} while(0)

// Đọc 4 byte (big-endian) thành số nguyên từ file nhị phân
int read_int(std::ifstream &ifs) {
    unsigned char bytes[4];
    ifs.read((char*)bytes, 4);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

// Đọc ảnh MNIST từ file .idx3-ubyte
std::vector<std::vector<float>> read_mnist_images(const std::string &filename) {
    std::ifstream ifs(filename, std::ios::binary);
    assert(ifs.is_open());

    int magic = read_int(ifs);
    int num_images = read_int(ifs);
    int num_rows = read_int(ifs);
    int num_cols = read_int(ifs);

    std::vector<std::vector<float>> images(num_images, std::vector<float>(num_rows * num_cols));
    for (int i = 0; i < num_images; ++i) {
        for (int j = 0; j < num_rows * num_cols; ++j) {
            unsigned char pixel;
            ifs.read((char*)&pixel, 1);
            images[i][j] = pixel / 255.0f;
        }
    }
    return images;
}

// Đọc nhãn MNIST từ file .idx1-ubyte
std::vector<int> read_mnist_labels(const std::string &filename) {
    std::ifstream ifs(filename, std::ios::binary);
    assert(ifs.is_open());

    int magic = read_int(ifs);
    int num_labels = read_int(ifs);

    std::vector<int> labels(num_labels);
    for (int i = 0; i < num_labels; ++i) {
        unsigned char label;
        ifs.read((char*)&label, 1);
        labels[i] = (int)label;
    }
    return labels;
}

// Kernel CUDA cho forward pass layer (ReLU activation)
__global__ void forward_kernel(float* input, float* weights, float* bias, float* output,
                               int batch_size, int input_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= output_size * batch_size) return;
    
    int output_idx = idx % output_size;
    int batch_idx = idx / output_size;
    
    float sum = bias[output_idx];
    for (int i = 0; i < input_size; i++) {
        sum += weights[output_idx * input_size + i] * input[batch_idx * input_size + i];
    }
    
    // ReLU activation
    output[idx] = fmaxf(0.0f, sum);
}

// Kernel CUDA cho softmax
__global__ void softmax_kernel(float* input, float* output, int batch_size, int size) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    // Tìm max để tránh overflow
    float max_val = -INFINITY;
    for (int i = 0; i < size; i++) {
        max_val = fmaxf(max_val, input[batch_idx * size + i]);
    }
    
    // Tính exp và tổng
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        output[batch_idx * size + i] = expf(input[batch_idx * size + i] - max_val);
        sum += output[batch_idx * size + i];
    }
    
    // Chuẩn hóa
    for (int i = 0; i < size; i++) {
        output[batch_idx * size + i] /= sum;
    }
}

__global__ void softmax_cross_entropy_grad_kernel(float* probs, int* labels, float* grad,
                                                 int batch_size, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * num_classes) return;
    
    int class_idx = idx % num_classes;
    int batch_idx = idx / num_classes;
    
    grad[idx] = probs[idx] - (class_idx == labels[batch_idx] ? 1.0f : 0.0f);
}

__global__ void backward_kernel(float* grad_output, float* output, float* input,
                               float* weights, float* bias, float* grad_input,
                               float* weight_updates, float* bias_updates,
                               int batch_size, int input_size, int output_size, float lr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= output_size) return;
    
    bias_updates[idx] = 0.0f;
    
    for (int b = 0; b < batch_size; b++) {
        float grad = grad_output[b * output_size + idx] * (output[b * output_size + idx] > 0 ? 1.0f : 0.0f);
        bias_updates[idx] += grad;
        
        for (int i = 0; i < input_size; i++) {
            atomicAdd(&weight_updates[idx * input_size + i], grad * input[b * input_size + i]);
            atomicAdd(&grad_input[b * input_size + i], grad * weights[idx * input_size + i]);
        }
    }
}

__global__ void update_weights_kernel(float* weights, float* bias, 
                                     float* weight_updates, float* bias_updates,
                                     int input_size, int output_size, float lr, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= output_size * input_size) return;
    
    int output_idx = idx / input_size;
    
    weights[idx] -= lr * weight_updates[idx] / batch_size;
    weight_updates[idx] = 0.0f;
    
    if (idx % input_size == 0) {
        bias[output_idx] -= lr * bias_updates[output_idx] / batch_size;
        bias_updates[output_idx] = 0.0f;
    }
}

// Lớp Layer sử dụng CUDA
class CudaLayer {
public:
    int input_size, output_size;
    float *d_weights, *d_bias;
    float *d_weight_updates, *d_bias_updates;
    float *d_input, *d_output, *d_grad_input;
    
    CudaLayer(int input_size, int output_size) : 
        input_size(input_size), output_size(output_size) {
        
        // Khởi tạo trọng số và bias trên GPU
        CUDA_CHECK(cudaMalloc(&d_weights, output_size * input_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_bias, output_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_weight_updates, output_size * input_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_bias_updates, output_size * sizeof(float)));
        
        // Tạo array tạm thời để khởi tạo trọng số
        std::vector<float> h_weights(output_size * input_size);
        std::vector<float> h_bias(output_size, 0.0f);
        
        // Khởi tạo trọng số với phân phối chuẩn
        std::mt19937 gen(std::random_device{}());
        std::normal_distribution<float> dist(0, 1.0f / sqrt(input_size));
        
        for (int i = 0; i < output_size * input_size; ++i) {
            h_weights[i] = dist(gen);
        }
        
        // Copy khởi tạo từ host sang device
        CUDA_CHECK(cudaMemcpy(d_weights, h_weights.data(), h_weights.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), h_bias.size() * sizeof(float), cudaMemcpyHostToDevice));
        
        // Đặt updates thành 0
        CUDA_CHECK(cudaMemset(d_weight_updates, 0, output_size * input_size * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_bias_updates, 0, output_size * sizeof(float)));
    }
    
    ~CudaLayer() {
        cudaFree(d_weights);
        cudaFree(d_bias);
        cudaFree(d_weight_updates);
        cudaFree(d_bias_updates);
    }
    
    void allocate_io_memory(int batch_size) {
        CUDA_CHECK(cudaMalloc(&d_input, batch_size * input_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, batch_size * output_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_input, batch_size * input_size * sizeof(float)));
    }
    
    void free_io_memory() {
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_grad_input);
    }
    
    float* forward(float* input, int batch_size) {
        // Lưu đầu vào để sử dụng trong backward
        CUDA_CHECK(cudaMemcpy(d_input, input, batch_size * input_size * sizeof(float), cudaMemcpyDeviceToDevice));
        
        // Kernel call cho forward pass
        int block_size = 256;
        int num_blocks = (batch_size * output_size + block_size - 1) / block_size;
        forward_kernel<<<num_blocks, block_size>>>(d_input, d_weights, d_bias, d_output, batch_size, input_size, output_size);
        CUDA_CHECK(cudaGetLastError());
        
        return d_output;
    }
    
    float* backward(float* grad_output, float lr, int batch_size) {
        // Đặt gradient đầu vào thành 0
        CUDA_CHECK(cudaMemset(d_grad_input, 0, batch_size * input_size * sizeof(float)));
        
        // Kernel call cho backward pass
        int block_size = 256;
        int num_blocks = (output_size + block_size - 1) / block_size;
        backward_kernel<<<num_blocks, block_size>>>(grad_output, d_output, d_input, d_weights, d_bias,
                                                  d_grad_input, d_weight_updates, d_bias_updates,
                                                  batch_size, input_size, output_size, lr);
        CUDA_CHECK(cudaGetLastError());
        
        // Cập nhật trọng số và bias
        num_blocks = (output_size * input_size + block_size - 1) / block_size;
        update_weights_kernel<<<num_blocks, block_size>>>(d_weights, d_bias, d_weight_updates, d_bias_updates,
                                                         input_size, output_size, lr, batch_size);
        CUDA_CHECK(cudaGetLastError());
        
        return d_grad_input;
    }
};

// Hàm tính cross entropy loss trên CPU (cho đơn giản)
float cross_entropy(const std::vector<float>& probs, int label) {
    return -std::log(std::max(probs[label], 1e-9f));
}

// Hàm chính
int main() {
    // Đọc dữ liệu ảnh và nhãn từ file
    auto train_images = read_mnist_images("train-images.idx3-ubyte");
    auto train_labels = read_mnist_labels("train-labels.idx1-ubyte");
    auto test_images = read_mnist_images("t10k-images.idx3-ubyte");
    auto test_labels = read_mnist_labels("t10k-labels.idx1-ubyte");
    
    int num_train = train_images.size();
    int num_test = test_images.size();
    int input_size = train_images[0].size();  // 784 (28x28)
    
    // Tạo các layer CUDA
    CudaLayer l1(784, 128);
    CudaLayer l2(128, 64);
    CudaLayer l3(64, 10);
    
    // Tham số huấn luyện
    float lr = 0.01f;  // Learning rate
    int epochs = 20;   // Số epoch
    int batch_size = 100; // Kích thước batch
    int num_batches = (num_train + batch_size - 1) / batch_size;
    
    // Cấp phát bộ nhớ cho batch và softmax
    float *d_batch_input, *d_batch_probs, *d_batch_output;
    int *d_batch_labels;
    float *d_grad_output;
    
    CUDA_CHECK(cudaMalloc(&d_batch_input, batch_size * input_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_batch_probs, batch_size * 10 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_batch_output, batch_size * 10 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_batch_labels, batch_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_grad_output, batch_size * 10 * sizeof(float)));
    
    // Cấp phát bộ nhớ I/O cho các layer
    l1.allocate_io_memory(batch_size);
    l2.allocate_io_memory(batch_size);
    l3.allocate_io_memory(batch_size);
    
    // Chuẩn bị file CSV output
    std::ofstream csv_file("training_results_cuda.csv");
    csv_file << "Epoch,Time (seconds),Loss,Accuracy\n";
    
    // Bắt đầu huấn luyện
    for (int epoch = 0; epoch < epochs; ++epoch) {
        auto start_time = std::chrono::high_resolution_clock::now();
        float total_loss = 0.0f;
        int total_correct = 0;
        
        // Shuffle dữ liệu
        std::vector<int> indices(num_train);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});
        
        // Huấn luyện theo batch
        for (int batch = 0; batch < num_batches; ++batch) {
            int start_idx = batch * batch_size;
            int end_idx = std::min(start_idx + batch_size, num_train);
            int current_batch_size = end_idx - start_idx;
            
            // Chuẩn bị batch dữ liệu
            std::vector<float> batch_data(current_batch_size * input_size);
            std::vector<int> batch_labels(current_batch_size);
            
            for (int i = 0; i < current_batch_size; ++i) {
                int idx = indices[start_idx + i];
                std::copy(train_images[idx].begin(), train_images[idx].end(), batch_data.begin() + i * input_size);
                batch_labels[i] = train_labels[idx];
            }
            
            // Copy batch lên GPU
            CUDA_CHECK(cudaMemcpy(d_batch_input, batch_data.data(), current_batch_size * input_size * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_batch_labels, batch_labels.data(), current_batch_size * sizeof(int), cudaMemcpyHostToDevice));
            
            // Forward pass
            float* out1 = l1.forward(d_batch_input, current_batch_size);
            float* out2 = l2.forward(out1, current_batch_size);
            float* out3 = l3.forward(out2, current_batch_size);
            
            // Softmax
            int block_size = 256;
            int num_blocks = (current_batch_size + block_size - 1) / block_size;
            softmax_kernel<<<num_blocks, block_size>>>(out3, d_batch_probs, current_batch_size, 10);
            CUDA_CHECK(cudaGetLastError());
            
            // Tính loss và độ chính xác (tải về CPU)
            std::vector<float> h_probs(current_batch_size * 10);
            CUDA_CHECK(cudaMemcpy(h_probs.data(), d_batch_probs, current_batch_size * 10 * sizeof(float), cudaMemcpyDeviceToHost));
            
            for (int i = 0; i < current_batch_size; ++i) {
                std::vector<float> sample_probs(h_probs.begin() + i * 10, h_probs.begin() + (i + 1) * 10);
                total_loss += cross_entropy(sample_probs, batch_labels[i]);
                
                int pred = std::distance(sample_probs.begin(), std::max_element(sample_probs.begin(), sample_probs.end()));
                if (pred == batch_labels[i]) total_correct++;
            }
            
            // Tính gradient của softmax với cross entropy (softmax_probs - one_hot_labels)
            softmax_cross_entropy_grad_kernel<<<(current_batch_size * 10 + block_size - 1) / block_size, block_size>>>(
                d_batch_probs, d_batch_labels, d_grad_output, current_batch_size, 10);
            CUDA_CHECK(cudaGetLastError());
            
            // Backward pass
            float* grad2 = l3.backward(d_grad_output, lr, current_batch_size);
            float* grad1 = l2.backward(grad2, lr, current_batch_size);
            l1.backward(grad1, lr, current_batch_size);
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> duration = end_time - start_time;
        float accuracy = 100.0f * total_correct / num_train;
        float avg_loss = total_loss / num_train;
        
        // Lưu kết quả
        csv_file << epoch + 1 << "," << duration.count() << "," << avg_loss << "," << accuracy << "\n";
        std::cout << "Epoch " << epoch + 1 << " | Loss: " << avg_loss
                  << " | Accuracy: " << accuracy << "% | Time: " << duration.count() << " seconds\n";
    }
    
    // Đánh giá trên tập test
    int test_correct = 0;
    for (int batch = 0; batch < (num_test + batch_size - 1) / batch_size; ++batch) {
        int start_idx = batch * batch_size;
        int end_idx = std::min(start_idx + batch_size, num_test);
        int current_batch_size = end_idx - start_idx;
        
        // Chuẩn bị batch dữ liệu test
        std::vector<float> batch_data(current_batch_size * input_size);
        std::vector<int> batch_labels(current_batch_size);
        
        for (int i = 0; i < current_batch_size; ++i) {
            std::copy(test_images[start_idx + i].begin(), test_images[start_idx + i].end(), batch_data.begin() + i * input_size);
            batch_labels[i] = test_labels[start_idx + i];
        }
        
        // Copy batch lên GPU
        CUDA_CHECK(cudaMemcpy(d_batch_input, batch_data.data(), current_batch_size * input_size * sizeof(float), cudaMemcpyHostToDevice));
        
        // Forward pass
        float* out1 = l1.forward(d_batch_input, current_batch_size);
        float* out2 = l2.forward(out1, current_batch_size);
        float* out3 = l3.forward(out2, current_batch_size);
        
        // Softmax
        int block_size = 256;
        int num_blocks = (current_batch_size + block_size - 1) / block_size;
        softmax_kernel<<<num_blocks, block_size>>>(out3, d_batch_probs, current_batch_size, 10);
        CUDA_CHECK(cudaGetLastError());
        
        // Tải kết quả về CPU và kiểm tra
        std::vector<float> h_probs(current_batch_size * 10);
        CUDA_CHECK(cudaMemcpy(h_probs.data(), d_batch_probs, current_batch_size * 10 * sizeof(float), cudaMemcpyDeviceToHost));
        
        for (int i = 0; i < current_batch_size; ++i) {
            std::vector<float> sample_probs(h_probs.begin() + i * 10, h_probs.begin() + (i + 1) * 10);
            int pred = std::distance(sample_probs.begin(), std::max_element(sample_probs.begin(), sample_probs.end()));
            if (pred == batch_labels[i]) test_correct++;
        }
    }
    
    float test_accuracy = 100.0f * test_correct / num_test;
    std::cout << "Test Accuracy: " << test_accuracy << "%" << std::endl;
    
    // In ra dự đoán của 10 ảnh đầu tiên
    std::cout << "\nDu doan 10 anh dau tien trong tap test\n";
    std::vector<float> first_10_data(10 * input_size);
    for (int i = 0; i < 10; ++i) {
        std::copy(test_images[i].begin(), test_images[i].end(), first_10_data.begin() + i * input_size);
    }
    
    CUDA_CHECK(cudaMemcpy(d_batch_input, first_10_data.data(), 10 * input_size * sizeof(float), cudaMemcpyHostToDevice));
    
    float* out1 = l1.forward(d_batch_input, 10);
    float* out2 = l2.forward(out1, 10);
    float* out3 = l3.forward(out2, 10);
    
    softmax_kernel<<<1, 256>>>(out3, d_batch_probs, 10, 10);
    CUDA_CHECK(cudaGetLastError());
    
    std::vector<float> h_probs(10 * 10);
    CUDA_CHECK(cudaMemcpy(h_probs.data(), d_batch_probs, 10 * 10 * sizeof(float), cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < 10; ++i) {
        std::vector<float> sample_probs(h_probs.begin() + i * 10, h_probs.begin() + (i + 1) * 10);
        int pred = std::distance(sample_probs.begin(), std::max_element(sample_probs.begin(), sample_probs.end()));
        std::cout << "Anh thu " << i + 1 << ": Nhan thuc te = " << test_labels[i] << ", Du doan = " << pred << "\n";
    }
    
    // Giải phóng bộ nhớ
    l1.free_io_memory();
    l2.free_io_memory();
    l3.free_io_memory();
    
    cudaFree(d_batch_input);
    cudaFree(d_batch_probs);
    cudaFree(d_batch_output);
    cudaFree(d_batch_labels);
    cudaFree(d_grad_output);
    
    csv_file.close();
    std::cout << "Da luu ket qua vao training_results_cuda.csv\n";
    
    return 0;
}