#include "gpu_check.h"
#include "mnist_loader.h"
#include "pcn/network.h"
#include "ui/app_window.h"

#include <yaml-cpp/yaml.h>
#include <cstdio>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <functional>

static std::vector<NetworkConfig_UI> loadConfigs(const std::string& path) {
    std::vector<NetworkConfig_UI> configs;
    YAML::Node root = YAML::LoadFile(path);

    if (!root["networks"]) {
        fprintf(stderr, "ERROR: No 'networks' key in config.yaml\n");
        return configs;
    }

    for (const auto& net : root["networks"]) {
        NetworkConfig_UI cfg;
        cfg.name = net["name"].as<std::string>();
        cfg.layers = net["layers"].as<std::vector<int>>();
        cfg.learning_rate = net["learning_rate"].as<float>();
        cfg.inference_steps = net["inference_steps"].as<int>();
        cfg.epochs = net["epochs"].as<int>();
        configs.push_back(cfg);
    }

    return configs;
}

int main(int argc, char** argv) {
    // 1. GPU detection
    GPUInfo gpu;
    if (!detectAndSelectGPU(gpu)) {
        return 1;
    }

    // 2. Load config
    std::string config_path = "config.yaml";
    if (argc > 1) config_path = argv[1];

    auto configs = loadConfigs(config_path);
    if (configs.empty()) {
        fprintf(stderr, "ERROR: No network configurations loaded.\n");
        return 1;
    }
    printf("Loaded %zu network configuration(s).\n", configs.size());

    // 3. Load MNIST
    MnistDataset mnist;
    if (!loadMnist("data", mnist)) {
        fprintf(stderr, "ERROR: Failed to load MNIST dataset.\n");
        return 1;
    }

    // 4. Setup shared training state
    TrainingState state;
    state.configs = configs;
    for (auto& cfg : configs) {
        NetworkResult_UI r;
        r.name = cfg.name;
        r.accuracy = 0.0f;
        r.per_class_correct.resize(10, 0);
        r.per_class_total.resize(10, 0);
        r.status = NetworkStatus::Pending;
        state.results.push_back(r);
    }

    // Training thread handle
    std::thread train_thread;

    // Start training callback — launched from UI thread
    auto start_training = [&](TrainingState& st) {
        if (st.running.load()) return;

        // Reset state
        st.stop_requested.store(false);
        st.running.store(true);
        {
            std::lock_guard<std::mutex> lock(st.history_mutex);
            st.energy_history.clear();
        }
        for (size_t i = 0; i < st.results.size(); i++) {
            std::lock_guard<std::mutex> lock(st.results_mutex);
            st.results[i].status = NetworkStatus::Pending;
            st.results[i].accuracy = 0.0f;
            std::fill(st.results[i].per_class_correct.begin(),
                      st.results[i].per_class_correct.end(), 0);
            std::fill(st.results[i].per_class_total.begin(),
                      st.results[i].per_class_total.end(), 0);
        }

        // Join previous thread if any
        if (train_thread.joinable()) {
            train_thread.join();
        }

        train_thread = std::thread([&st, &mnist]() {
            for (size_t i = 0; i < st.configs.size(); i++) {
                if (st.stop_requested.load()) break;

                auto& cfg = st.configs[i];

                // Update status
                st.current_network.store(static_cast<int>(i));
                st.current_network_name = cfg.name;
                st.current_epoch.store(0);
                st.current_sample.store(0);
                st.total_epochs.store(cfg.epochs);
                st.total_samples.store(mnist.num_train);

                {
                    std::lock_guard<std::mutex> lock(st.results_mutex);
                    st.results[i].status = NetworkStatus::Training;
                }
                {
                    std::lock_guard<std::mutex> lock(st.history_mutex);
                    st.energy_history.clear();
                }

                // Init network
                PCNNetwork net;
                net.init(cfg.layers, cfg.learning_rate, cfg.inference_steps);

                // Set callback
                net.set_callback([&st](int epoch, int sample, float energy) {
                    st.current_epoch.store(epoch);
                    st.current_sample.store(sample);
                    st.current_energy.store(energy);

                    // Store energy history (downsample to avoid memory bloat)
                    if (sample % 100 == 0) {
                        std::lock_guard<std::mutex> lock(st.history_mutex);
                        st.energy_history.push_back(energy);
                    }
                });

                // Train
                net.train(mnist.train_images.data(), mnist.train_labels.data(),
                          mnist.num_train, cfg.epochs, st.stop_requested);

                if (st.stop_requested.load()) {
                    std::lock_guard<std::mutex> lock(st.results_mutex);
                    st.results[i].status = NetworkStatus::Error;
                    net.cleanup();
                    break;
                }

                // Evaluate with per-class breakdown
                std::vector<int> per_class_correct, per_class_total;
                float accuracy = net.evaluate_per_class(
                    mnist.test_images.data(), mnist.test_labels.data(),
                    mnist.num_test, per_class_correct, per_class_total);

                {
                    std::lock_guard<std::mutex> lock(st.results_mutex);
                    st.results[i].accuracy = accuracy;
                    st.results[i].per_class_correct = per_class_correct;
                    st.results[i].per_class_total = per_class_total;
                    st.results[i].status = NetworkStatus::Done;
                }

                net.cleanup();
            }

            st.running.store(false);
        });
    };

    // 5. Launch UI (blocking main loop)
    runAppWindow(gpu.name, state, start_training);

    // Wait for training thread to finish
    state.stop_requested.store(true);
    if (train_thread.joinable()) {
        train_thread.join();
    }

    return 0;
}
