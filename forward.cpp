// MLP don gian nhan dien chu so tu MNIST
// Code da duoc viet lai ro rang, chia ham va comment tieng Viet

#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <chrono>
#include <sstream>

// ============================ HAM HO TRO DOC DU LIEU ==============================

// Doc 4 byte (big-endian) thanh so nguyen tu file nhi phan
int read_int(std::ifstream &ifs) {
    unsigned char bytes[4];
    ifs.read((char*)bytes, 4);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

// Doc anh MNIST tu file .idx3-ubyte
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

// Doc nhan MNIST tu file .idx1-ubyte
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

///  Lop MLP (Multi-Layer Perceptron)
struct Layer {
    std::vector<std::vector<float>> weights;
    std::vector<float> biases;
    std::vector<float> outputs;
    std::vector<float> inputs;
    std::vector<float> deltas;

    Layer(int input_size, int output_size) {
        std::mt19937 gen(std::random_device{}());
        std::normal_distribution<float> dist(0, 1.0f / sqrt(input_size));

        weights.resize(output_size, std::vector<float>(input_size));
        biases.resize(output_size, 0.0f);
        outputs.resize(output_size);
        deltas.resize(output_size);

        for (auto &row : weights)
            for (auto &w : row)
                w = dist(gen);
    }

    // Lan truyen thuan (forward) song song bang OpenMP
    std::vector<float> forward(const std::vector<float> &input) {
        inputs = input;
        outputs.resize(weights.size());

        #pragma omp parallel for
        for (size_t i = 0; i < weights.size(); ++i) {
            float z = biases[i];
            for (size_t j = 0; j < weights[0].size(); ++j) {
                z += weights[i][j] * input[j];
            }
            outputs[i] = std::max(0.0f, z); // Kich hoat ReLU
        }
        return outputs;
}


    // Lan truyen nguoc (backward)
    std::vector<float> backward(const std::vector<float> &grad_output, float lr) {
        std::vector<float> grad_input(weights[0].size(), 0.0f);

        for (size_t i = 0; i < weights.size(); ++i) {
            float grad = grad_output[i] * (outputs[i] > 0 ? 1.0f : 0.0f); // Dao ham ReLU
            deltas[i] = grad;

            for (size_t j = 0; j < weights[0].size(); ++j) {
                grad_input[j] += grad * weights[i][j];
                weights[i][j] -= lr * grad * inputs[j];
            }
            biases[i] -= lr * grad;
        }

        return grad_input;
    }
};

// Ham softmax bien vector thanh xac suat
std::vector<float> softmax(const std::vector<float>& x) {
    float max_x = *std::max_element(x.begin(), x.end());
    std::vector<float> exps(x.size());
    float sum = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        exps[i] = std::exp(x[i] - max_x);
        sum += exps[i];
    }
    for (float &v : exps) v /= sum;
    return exps;
}

// Ham tinh cross entropy loss giua xac suat va nhan dung
float cross_entropy(const std::vector<float>& probs, int label) {
    return -std::log(std::max(probs[label], 1e-9f));
}


int main() {
    // Doc du lieu anh va nhan tu file
    auto train_images = read_mnist_images("train-images-idx3-ubyte/train-images.idx3-ubyte");
    auto train_labels = read_mnist_labels("train-labels-idx1-ubyte/train-labels.idx1-ubyte");

    // Khoi tao mang MLP 3 lop
    Layer l1(784, 128);
    Layer l2(128, 64);
    Layer l3(64, 10);

    float lr = 0.01f; // Learning rate
    int epochs = 10;  // So epoch

    std::ofstream csv_file("training_results.csv");
    csv_file << "Epoch,Time (seconds),Loss,Accuracy\n";

    for (int epoch = 0; epoch < epochs; ++epoch) {
        auto start_time = std::chrono::high_resolution_clock::now();
        float loss = 0;
        int correct = 0;

        for (size_t i = 0; i < train_images.size(); ++i) {
            auto x = train_images[i];
            int y = train_labels[i];

            // Lan truyen tien
            auto out1 = l1.forward(x);
            auto out2 = l2.forward(out1);
            auto out3 = l3.forward(out2);
            auto probs = softmax(out3);

            // Tinh loss va do chinh xac
            loss += cross_entropy(probs, y);
            correct += (std::distance(probs.begin(), std::max_element(probs.begin(), probs.end())) == y);

            // Lan truyen nguoc
            std::vector<float> grad(10);
            for (int j = 0; j < 10; ++j)
                grad[j] = probs[j] - (j == y ? 1.0f : 0.0f);

            auto g2 = l3.backward(grad, lr);
            auto g1 = l2.backward(g2, lr);
            l1.backward(g1, lr);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> duration = end_time - start_time;
        float accuracy = 100.0f * correct / train_images.size();

        csv_file << epoch + 1 << "," << duration.count() << "," << loss / train_images.size() << "," << accuracy << "\n";
        std::cout << "Epoch " << epoch + 1 << " | Loss: " << loss / train_images.size()
                  << " | Accuracy: " << accuracy << "% | Time: " << duration.count() << " seconds\n";
    }

        // Đọc tập test
        auto test_images = read_mnist_images("t10k-images-idx3-ubyte/t10k-images.idx3-ubyte");
        auto test_labels = read_mnist_labels("t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte");
    
        int test_correct = 0;
        for (size_t i = 0; i < test_images.size(); ++i) {
            auto x = test_images[i];
            int y = test_labels[i];
    
            auto out1 = l1.forward(x);
            auto out2 = l2.forward(out1);
            auto out3 = l3.forward(out2);
            auto probs = softmax(out3);
    
            int pred = std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));
            if (pred == y) test_correct++;
        }
    
        float test_accuracy = 100.0f * test_correct / test_images.size();
        std::cout << "Test Accuracy: " << test_accuracy << "%" << std::endl;
    
        // In ra dự đoán của 10 ảnh đầu tiên
    std::cout << "\n=== Du doan 10 anh dau tien trong tap test ===\n";
    for (int i = 0; i < 10; ++i) {
        auto x = test_images[i];
        int y = test_labels[i];

        auto out1 = l1.forward(x);
        auto out2 = l2.forward(out1);
        auto out3 = l3.forward(out2);
        auto probs = softmax(out3);

        int pred = std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));

        std::cout << "Anh thu " << i + 1 << ": Nhan thuc te = " << y << ", Du doan = " << pred << "\n";
}


    csv_file.close();
    std::cout << "Da luu ket qua vao training_results.csv\n";
    return 0;
}
