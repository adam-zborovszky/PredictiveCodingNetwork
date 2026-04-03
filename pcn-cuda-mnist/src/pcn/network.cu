#include "pcn/network.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <random>
#include <numeric>
#include <atomic>

// ─── Helper macros ──────────────────────────────────────────────────────────

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

static constexpr int BLOCK_SIZE = 256;

static int gridSize(int n) {
    return (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

// ─── CUDA Kernels ───────────────────────────────────────────────────────────

// Matrix-vector multiply: out[i] = sum_j( W[i * cols + j] * x[j] ) + bias[i]
// Then apply tanh activation.
__global__ void forward_predict_kernel(const float* __restrict__ weights,
                                       const float* __restrict__ mu_above,
                                       const float* __restrict__ bias,
                                       float* __restrict__ prediction,
                                       int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return;

    float sum = bias[i];
    for (int j = 0; j < cols; j++) {
        sum += weights[i * cols + j] * mu_above[j];
    }
    prediction[i] = tanhf(sum);
}

// Softmax forward prediction for output layer predicting layer below.
// Two-pass: first compute max, then exp and normalize.
// This kernel is launched with a single block since output is small (10).
__global__ void forward_predict_softmax_kernel(const float* __restrict__ weights,
                                                const float* __restrict__ mu_above,
                                                const float* __restrict__ bias,
                                                float* __restrict__ prediction,
                                                int rows, int cols) {
    // Use shared memory for the linear outputs
    extern __shared__ float shared[];
    float* linear = shared;           // [rows]
    float* reduction = shared + rows; // [rows] for max/sum

    int tid = threadIdx.x;
    if (tid >= rows) return;

    // Compute linear output
    float sum = bias[tid];
    for (int j = 0; j < cols; j++) {
        sum += weights[tid * cols + j] * mu_above[j];
    }
    linear[tid] = sum;
    __syncthreads();

    // Find max (simple reduction for small arrays)
    reduction[tid] = linear[tid];
    __syncthreads();
    for (int stride = 1; stride < rows; stride *= 2) {
        if (tid % (2 * stride) == 0 && tid + stride < rows) {
            reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
        }
        __syncthreads();
    }
    float max_val = reduction[0];
    __syncthreads();

    // Compute exp(x - max)
    float exp_val = expf(linear[tid] - max_val);
    reduction[tid] = exp_val;
    __syncthreads();

    // Sum reduction
    for (int stride = 1; stride < rows; stride *= 2) {
        if (tid % (2 * stride) == 0 && tid + stride < rows) {
            reduction[tid] += reduction[tid + stride];
        }
        __syncthreads();
    }
    float total = reduction[0];
    __syncthreads();

    prediction[tid] = exp_val / total;
}

// epsilon_l = mu_l - prediction_l
__global__ void compute_error_kernel(const float* __restrict__ mu,
                                     const float* __restrict__ prediction,
                                     float* __restrict__ epsilon,
                                     int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    epsilon[i] = mu[i] - prediction[i];
}

// Update mu for hidden layers:
// mu_l += lr_infer * (-epsilon_l + W_{l+1}^T * epsilon_{l+1})
__global__ void update_mu_kernel(float* __restrict__ mu,
                                 const float* __restrict__ epsilon,
                                 const float* __restrict__ weights_above,
                                 const float* __restrict__ epsilon_above,
                                 float lr_infer,
                                 int size,
                                 int size_above) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    // W_{l+1}^T * epsilon_{l+1}: column i of weights_above dotted with epsilon_above
    // weights_above is [size_above x size], so W^T[i,j] = W[j,i] = weights_above[j * size + i]
    float top_down = 0.0f;
    for (int j = 0; j < size_above; j++) {
        top_down += weights_above[j * size + i] * epsilon_above[j];
    }

    mu[i] += lr_infer * (-epsilon[i] + top_down);
}

// Weight update: W_l += lr * epsilon_l * mu_{l-1}^T
__global__ void update_weights_kernel(float* __restrict__ weights,
                                      const float* __restrict__ epsilon,
                                      const float* __restrict__ mu_below,
                                      float lr,
                                      int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total) return;

    int i = idx / cols;
    int j = idx % cols;

    weights[idx] += lr * epsilon[i] * mu_below[j];
}

// Bias update: b_l += lr * epsilon_l
__global__ void update_biases_kernel(float* __restrict__ biases,
                                     const float* __restrict__ epsilon,
                                     float lr,
                                     int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    biases[i] += lr * epsilon[i];
}

// Copy data to device memory
__global__ void copy_kernel(float* __restrict__ dst,
                            const float* __restrict__ src,
                            int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    dst[i] = src[i];
}

// Set one-hot vector
__global__ void set_onehot_kernel(float* __restrict__ dst, int label, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    dst[i] = (i == label) ? 1.0f : 0.0f;
}

// Compute squared error sum (for energy): sum of epsilon^2
__global__ void squared_error_kernel(const float* __restrict__ epsilon,
                                     float* __restrict__ partial_sum,
                                     int size) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < size) ? epsilon[i] * epsilon[i] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sum[blockIdx.x] = sdata[0];
    }
}

// ─── Host helper: compute total energy ──────────────────────────────────────

static float computeEnergy(const std::vector<PCNLayer>& layers) {
    float total = 0.0f;
    for (size_t l = 1; l < layers.size(); l++) {
        int size = layers[l].size;
        int nblocks = gridSize(size);

        float* d_partial;
        CUDA_CHECK(cudaMalloc(&d_partial, nblocks * sizeof(float)));

        squared_error_kernel<<<nblocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
            layers[l].epsilon, d_partial, size);

        std::vector<float> h_partial(nblocks);
        CUDA_CHECK(cudaMemcpy(h_partial.data(), d_partial,
                              nblocks * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_partial));

        for (int b = 0; b < nblocks; b++) {
            total += h_partial[b];
        }
    }
    return total;
}

// ─── Temporary device buffer for predictions ────────────────────────────────

struct PredictionBuffers {
    std::vector<float*> pred;  // one buffer per layer (except layer 0)

    void allocate(const std::vector<PCNLayer>& layers) {
        pred.resize(layers.size(), nullptr);
        for (size_t l = 0; l < layers.size(); l++) {
            CUDA_CHECK(cudaMalloc(&pred[l], layers[l].size * sizeof(float)));
        }
    }

    void free() {
        for (auto p : pred) {
            if (p) cudaFree(p);
        }
        pred.clear();
    }
};

// ─── PCNNetwork implementation ──────────────────────────────────────────────

void PCNNetwork::init(const std::vector<int>& layer_sizes, float lr, int inf_steps) {
    learning_rate = lr;
    inference_steps = inf_steps;
    layers.resize(layer_sizes.size());

    std::mt19937 rng(12345);

    for (size_t l = 0; l < layer_sizes.size(); l++) {
        auto& layer = layers[l];
        layer.size = layer_sizes[l];
        layer.prev_size = (l > 0) ? layer_sizes[l - 1] : 0;

        // Allocate device memory for mu and epsilon
        CUDA_CHECK(cudaMalloc(&layer.mu, layer.size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&layer.epsilon, layer.size * sizeof(float)));
        CUDA_CHECK(cudaMemset(layer.mu, 0, layer.size * sizeof(float)));
        CUDA_CHECK(cudaMemset(layer.epsilon, 0, layer.size * sizeof(float)));

        if (l > 0) {
            int weight_count = layer.size * layer.prev_size;

            // Allocate weights and biases
            CUDA_CHECK(cudaMalloc(&layer.weights, weight_count * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&layer.biases, layer.size * sizeof(float)));

            // Xavier initialization: N(0, sqrt(2 / (fan_in + fan_out)))
            float stddev = sqrtf(2.0f / (layer.prev_size + layer.size));
            std::normal_distribution<float> dist(0.0f, stddev);

            std::vector<float> h_weights(weight_count);
            std::vector<float> h_biases(layer.size, 0.0f);

            for (int i = 0; i < weight_count; i++) {
                h_weights[i] = dist(rng);
            }

            CUDA_CHECK(cudaMemcpy(layer.weights, h_weights.data(),
                                  weight_count * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(layer.biases, h_biases.data(),
                                  layer.size * sizeof(float), cudaMemcpyHostToDevice));
        } else {
            layer.weights = nullptr;
            layer.biases = nullptr;
        }
    }

    printf("PCNNetwork initialized: ");
    for (size_t l = 0; l < layer_sizes.size(); l++) {
        printf("%d%s", layer_sizes[l], (l < layer_sizes.size() - 1) ? " -> " : "");
    }
    printf(" (lr=%.4f, inference_steps=%d)\n", lr, inf_steps);
}

void PCNNetwork::cleanup() {
    for (auto& layer : layers) {
        if (layer.mu) { cudaFree(layer.mu); layer.mu = nullptr; }
        if (layer.epsilon) { cudaFree(layer.epsilon); layer.epsilon = nullptr; }
        if (layer.weights) { cudaFree(layer.weights); layer.weights = nullptr; }
        if (layer.biases) { cudaFree(layer.biases); layer.biases = nullptr; }
    }
    layers.clear();
}

void PCNNetwork::train_sample(const float* d_image, int label) {
    int num_layers = static_cast<int>(layers.size());
    int input_size = layers[0].size;
    int output_size = layers[num_layers - 1].size;

    // Allocate prediction buffers
    PredictionBuffers bufs;
    bufs.allocate(layers);

    // 1. Clamp input layer mu to image
    CUDA_CHECK(cudaMemcpy(layers[0].mu, d_image, input_size * sizeof(float),
                          cudaMemcpyDeviceToDevice));

    // 2. Clamp output layer mu to one-hot(label)
    set_onehot_kernel<<<gridSize(output_size), BLOCK_SIZE>>>(
        layers[num_layers - 1].mu, label, output_size);

    // 3. Initialize hidden layer mus to small random values (or leave as previous)
    // We'll initialize them with a forward pass
    // Actually: just init to zero and let inference converge
    for (int l = 1; l < num_layers - 1; l++) {
        CUDA_CHECK(cudaMemset(layers[l].mu, 0, layers[l].size * sizeof(float)));
    }

    // Inference learning rate (typically larger than weight lr)
    float lr_infer = 0.1f;

    // 3. Run inference_steps iterations
    // Formulation: pred_l = f(W_l * mu_{l-1} + b_l), epsilon_l = mu_l - pred_l
    // W_l is [size_l x prev_size_l], mapping from layer below to current layer.
    for (int step = 0; step < inference_steps; step++) {
        // a. Forward predict for each layer l >= 1
        for (int l = 1; l < num_layers; l++) {
            int rows = layers[l].size;
            int cols = layers[l].prev_size;

            if (l == num_layers - 1) {
                // Output layer: use softmax
                int smem = 2 * rows * sizeof(float);
                forward_predict_softmax_kernel<<<1, rows, smem>>>(
                    layers[l].weights, layers[l - 1].mu, layers[l].biases,
                    bufs.pred[l], rows, cols);
            } else {
                forward_predict_kernel<<<gridSize(rows), BLOCK_SIZE>>>(
                    layers[l].weights, layers[l - 1].mu, layers[l].biases,
                    bufs.pred[l], rows, cols);
            }
        }

        // b. Compute errors: epsilon_l = mu_l - pred_l
        for (int l = 1; l < num_layers; l++) {
            compute_error_kernel<<<gridSize(layers[l].size), BLOCK_SIZE>>>(
                layers[l].mu, bufs.pred[l], layers[l].epsilon, layers[l].size);
        }

        // c. Update mu for hidden layers (not input, not output — those are clamped)
        for (int l = 1; l < num_layers - 1; l++) {
            update_mu_kernel<<<gridSize(layers[l].size), BLOCK_SIZE>>>(
                layers[l].mu, layers[l].epsilon,
                layers[l + 1].weights, layers[l + 1].epsilon,
                lr_infer, layers[l].size, layers[l + 1].size);
        }

        // Re-clamp input and output
        CUDA_CHECK(cudaMemcpy(layers[0].mu, d_image, input_size * sizeof(float),
                              cudaMemcpyDeviceToDevice));
        set_onehot_kernel<<<gridSize(output_size), BLOCK_SIZE>>>(
            layers[num_layers - 1].mu, label, output_size);
    }

    // 4. Update weights and biases for all layers with weights
    for (int l = 1; l < num_layers; l++) {
        int rows = layers[l].size;
        int cols = layers[l].prev_size;
        int total = rows * cols;

        update_weights_kernel<<<gridSize(total), BLOCK_SIZE>>>(
            layers[l].weights, layers[l].epsilon, layers[l - 1].mu,
            learning_rate, rows, cols);

        update_biases_kernel<<<gridSize(rows), BLOCK_SIZE>>>(
            layers[l].biases, layers[l].epsilon, learning_rate, rows);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    bufs.free();
}

int PCNNetwork::predict(const float* d_image) {
    int num_layers = static_cast<int>(layers.size());
    int input_size = layers[0].size;
    int output_size = layers[num_layers - 1].size;

    PredictionBuffers bufs;
    bufs.allocate(layers);

    // 1. Clamp input
    CUDA_CHECK(cudaMemcpy(layers[0].mu, d_image, input_size * sizeof(float),
                          cudaMemcpyDeviceToDevice));

    // Initialize hidden and output layers to zero
    for (int l = 1; l < num_layers; l++) {
        CUDA_CHECK(cudaMemset(layers[l].mu, 0, layers[l].size * sizeof(float)));
    }

    float lr_infer = 0.1f;

    // 2. Run inference (no label clamping on output)
    for (int step = 0; step < inference_steps; step++) {
        for (int l = 1; l < num_layers; l++) {
            int rows = layers[l].size;
            int cols = layers[l].prev_size;

            if (l == num_layers - 1) {
                int smem = 2 * rows * sizeof(float);
                forward_predict_softmax_kernel<<<1, rows, smem>>>(
                    layers[l].weights, layers[l - 1].mu, layers[l].biases,
                    bufs.pred[l], rows, cols);
            } else {
                forward_predict_kernel<<<gridSize(rows), BLOCK_SIZE>>>(
                    layers[l].weights, layers[l - 1].mu, layers[l].biases,
                    bufs.pred[l], rows, cols);
            }
        }

        for (int l = 1; l < num_layers; l++) {
            compute_error_kernel<<<gridSize(layers[l].size), BLOCK_SIZE>>>(
                layers[l].mu, bufs.pred[l], layers[l].epsilon, layers[l].size);
        }

        // Update mu for hidden layers AND output layer (output is free during prediction)
        for (int l = 1; l < num_layers - 1; l++) {
            update_mu_kernel<<<gridSize(layers[l].size), BLOCK_SIZE>>>(
                layers[l].mu, layers[l].epsilon,
                layers[l + 1].weights, layers[l + 1].epsilon,
                lr_infer, layers[l].size, layers[l + 1].size);
        }

        // Update output layer mu: only bottom-up error signal (no layer above)
        // mu_output += lr_infer * (-epsilon_output)
        // But actually, during prediction the output is free to settle.
        // We use the prediction directly: mu_output = pred_output
        // Simpler: just set output mu to the prediction (which is the generative model's guess)
        CUDA_CHECK(cudaMemcpy(layers[num_layers - 1].mu, bufs.pred[num_layers - 1],
                              output_size * sizeof(float), cudaMemcpyDeviceToDevice));

        // Re-clamp input
        CUDA_CHECK(cudaMemcpy(layers[0].mu, d_image, input_size * sizeof(float),
                              cudaMemcpyDeviceToDevice));
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // 3. Return argmax of output mu
    std::vector<float> h_output(output_size);
    CUDA_CHECK(cudaMemcpy(h_output.data(), layers[num_layers - 1].mu,
                          output_size * sizeof(float), cudaMemcpyDeviceToHost));

    bufs.free();

    return static_cast<int>(
        std::distance(h_output.begin(),
                      std::max_element(h_output.begin(), h_output.end())));
}

void PCNNetwork::train(const float* images, const int* labels, int count, int epochs,
                       std::atomic<bool>& stop_requested) {
    // Upload all images and labels to device
    float* d_images;
    int* d_labels;
    int image_size = layers[0].size;

    CUDA_CHECK(cudaMalloc(&d_images, count * image_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_labels, count * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_images, images, count * image_size * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_labels, labels, count * sizeof(int),
                          cudaMemcpyHostToDevice));

    // Create a shuffled index for each epoch
    std::vector<int> indices(count);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(42);

    for (int epoch = 0; epoch < epochs; epoch++) {
        if (stop_requested.load()) break;

        std::shuffle(indices.begin(), indices.end(), rng);

        for (int s = 0; s < count; s++) {
            if (stop_requested.load()) break;

            int idx = indices[s];
            float* sample_ptr = d_images + idx * image_size;

            // Read label from host (labels array)
            int lbl = labels[idx];

            train_sample(sample_ptr, lbl);

            // Compute energy and call callback
            if (callback) {
                float energy = computeEnergy(layers);
                callback(epoch, s, energy);
            }
        }
    }

    CUDA_CHECK(cudaFree(d_images));
    CUDA_CHECK(cudaFree(d_labels));
}

float PCNNetwork::evaluate(const float* images, const int* labels, int count) {
    int image_size = layers[0].size;

    // Upload images to device
    float* d_images;
    CUDA_CHECK(cudaMalloc(&d_images, count * image_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_images, images, count * image_size * sizeof(float),
                          cudaMemcpyHostToDevice));

    int correct = 0;
    for (int i = 0; i < count; i++) {
        int pred = predict(d_images + i * image_size);
        if (pred == labels[i]) correct++;
    }

    CUDA_CHECK(cudaFree(d_images));

    return static_cast<float>(correct) / static_cast<float>(count);
}

float PCNNetwork::evaluate_per_class(const float* images, const int* labels, int count,
                                     std::vector<int>& per_class_correct,
                                     std::vector<int>& per_class_total) {
    int image_size = layers[0].size;

    per_class_correct.assign(10, 0);
    per_class_total.assign(10, 0);

    float* d_images;
    CUDA_CHECK(cudaMalloc(&d_images, count * image_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_images, images, count * image_size * sizeof(float),
                          cudaMemcpyHostToDevice));

    int correct = 0;
    for (int i = 0; i < count; i++) {
        int label = labels[i];
        int pred = predict(d_images + i * image_size);
        per_class_total[label]++;
        if (pred == label) {
            per_class_correct[label]++;
            correct++;
        }
    }

    CUDA_CHECK(cudaFree(d_images));
    return static_cast<float>(correct) / static_cast<float>(count);
}

void PCNNetwork::set_callback(TrainingCallback cb) {
    callback = cb;
}
