#pragma once

#include <string>
#include <vector>
#include <atomic>
#include <mutex>
#include <functional>

enum class NetworkStatus { Pending, Training, Done, Error };

struct NetworkConfig_UI {
    std::string name;
    std::vector<int> layers;
    float learning_rate;
    int inference_steps;
    int epochs;
};

struct NetworkResult_UI {
    std::string name;
    float accuracy;
    std::vector<int> per_class_correct;
    std::vector<int> per_class_total;
    NetworkStatus status;
};

struct TrainingState {
    std::atomic<bool> running{false};
    std::atomic<bool> stop_requested{false};
    std::atomic<int> current_network{-1};
    std::atomic<int> current_epoch{0};
    std::atomic<int> current_sample{0};
    std::atomic<int> total_epochs{0};
    std::atomic<int> total_samples{0};
    std::atomic<float> current_energy{0.0f};
    std::string current_network_name;

    std::vector<float> energy_history;
    std::mutex history_mutex;

    std::vector<NetworkResult_UI> results;
    std::mutex results_mutex;

    std::vector<NetworkConfig_UI> configs;
};

// Callback type for starting training from UI
using StartTrainingFunc = std::function<void(TrainingState&)>;

// Initialize and run the ImGui application window (blocks until closed).
bool runAppWindow(const std::string& gpu_name, TrainingState& state,
                  StartTrainingFunc start_training);
