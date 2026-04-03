#pragma once

#include <vector>
#include <string>
#include <cstdint>

struct MnistDataset {
    std::vector<float> train_images;   // [num_train * 784]
    std::vector<int>   train_labels;   // [num_train]
    std::vector<float> test_images;    // [num_test * 784]
    std::vector<int>   test_labels;    // [num_test]
    int num_train;
    int num_test;
    int image_size;  // 784
};

// Downloads MNIST if needed and loads into MnistDataset.
// data_dir: directory to store/cache the IDX files.
// Returns true on success.
bool loadMnist(const std::string& data_dir, MnistDataset& dataset);
