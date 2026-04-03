#pragma once

#include <vector>
#include <functional>
#include <atomic>

struct PCNLayer {
    float* mu;          // neural activities [size] (device memory)
    float* epsilon;     // prediction errors [size] (device memory)
    float* weights;     // W [size x prev_size] (device memory)
    float* biases;      // b [size] (device memory)
    int size;
    int prev_size;
};

using TrainingCallback = std::function<void(int epoch, int sample, float energy)>;

struct PCNNetwork {
    std::vector<PCNLayer> layers;
    float learning_rate;
    int inference_steps;
    TrainingCallback callback;

    void init(const std::vector<int>& layer_sizes, float lr, int inf_steps);
    void cleanup();

    void train_sample(const float* image, int label);
    void train(const float* images, const int* labels, int count, int epochs,
               std::atomic<bool>& stop_requested);
    int predict(const float* image);
    float evaluate(const float* images, const int* labels, int count);
    float evaluate_per_class(const float* images, const int* labels, int count,
                             std::vector<int>& per_class_correct,
                             std::vector<int>& per_class_total);

    void set_callback(TrainingCallback cb);
};
