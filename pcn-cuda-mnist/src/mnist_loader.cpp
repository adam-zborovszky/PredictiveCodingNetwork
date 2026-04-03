#include "mnist_loader.h"
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <random>
#include <sys/stat.h>

static const char* MNIST_MIRROR = "https://storage.googleapis.com/cvdf-datasets/mnist/";

static const char* MNIST_FILES[] = {
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz"
};

static bool fileExists(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

static bool downloadFile(const std::string& url, const std::string& dest) {
    std::string cmd = "curl -sL \"" + url + "\" -o \"" + dest + "\"";
    int ret = system(cmd.c_str());
    return ret == 0;
}

static bool decompressGz(const std::string& gz_path) {
    std::string cmd = "gunzip -kf \"" + gz_path + "\"";
    int ret = system(cmd.c_str());
    return ret == 0;
}

static uint32_t readBigEndian32(std::ifstream& f) {
    uint8_t buf[4];
    f.read(reinterpret_cast<char*>(buf), 4);
    return (static_cast<uint32_t>(buf[0]) << 24) |
           (static_cast<uint32_t>(buf[1]) << 16) |
           (static_cast<uint32_t>(buf[2]) << 8)  |
           (static_cast<uint32_t>(buf[3]));
}

static bool loadImages(const std::string& path, std::vector<float>& images, int& count, int& image_size) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        fprintf(stderr, "ERROR: Cannot open %s\n", path.c_str());
        return false;
    }

    uint32_t magic = readBigEndian32(f);
    if (magic != 2051) {
        fprintf(stderr, "ERROR: Invalid image file magic: %u (expected 2051)\n", magic);
        return false;
    }

    count = static_cast<int>(readBigEndian32(f));
    int rows = static_cast<int>(readBigEndian32(f));
    int cols = static_cast<int>(readBigEndian32(f));
    image_size = rows * cols;

    images.resize(count * image_size);
    std::vector<uint8_t> raw(count * image_size);
    f.read(reinterpret_cast<char*>(raw.data()), raw.size());

    for (int i = 0; i < count * image_size; i++) {
        images[i] = static_cast<float>(raw[i]) / 255.0f;
    }

    return true;
}

static bool loadLabels(const std::string& path, std::vector<int>& labels, int& count) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        fprintf(stderr, "ERROR: Cannot open %s\n", path.c_str());
        return false;
    }

    uint32_t magic = readBigEndian32(f);
    if (magic != 2049) {
        fprintf(stderr, "ERROR: Invalid label file magic: %u (expected 2049)\n", magic);
        return false;
    }

    count = static_cast<int>(readBigEndian32(f));
    labels.resize(count);
    std::vector<uint8_t> raw(count);
    f.read(reinterpret_cast<char*>(raw.data()), raw.size());

    for (int i = 0; i < count; i++) {
        labels[i] = static_cast<int>(raw[i]);
    }

    return true;
}

bool loadMnist(const std::string& data_dir, MnistDataset& dataset) {
    // Create data directory
    std::string mkdir_cmd = "mkdir -p \"" + data_dir + "\"";
    system(mkdir_cmd.c_str());

    // Download and decompress if needed
    for (const char* filename : MNIST_FILES) {
        std::string gz_path = data_dir + "/" + filename;
        std::string raw_path = gz_path.substr(0, gz_path.size() - 3); // strip .gz

        if (!fileExists(raw_path)) {
            if (!fileExists(gz_path)) {
                printf("Downloading %s ...\n", filename);
                std::string url = std::string(MNIST_MIRROR) + filename;
                if (!downloadFile(url, gz_path)) {
                    fprintf(stderr, "ERROR: Failed to download %s\n", filename);
                    return false;
                }
            }
            printf("Decompressing %s ...\n", filename);
            if (!decompressGz(gz_path)) {
                fprintf(stderr, "ERROR: Failed to decompress %s\n", filename);
                return false;
            }
        }
    }

    // Load all images and labels
    std::vector<float> all_images;
    std::vector<int> all_labels;
    int img_count = 0, lbl_count = 0;
    int image_size = 0;

    // Train set
    std::vector<float> train_imgs, test_imgs_raw;
    std::vector<int> train_lbls, test_lbls_raw;
    int train_count = 0, test_count_raw = 0;
    int isz1 = 0, isz2 = 0;

    if (!loadImages(data_dir + "/train-images-idx3-ubyte", train_imgs, train_count, isz1))
        return false;
    if (!loadLabels(data_dir + "/train-labels-idx1-ubyte", train_lbls, img_count))
        return false;
    if (!loadImages(data_dir + "/t10k-images-idx3-ubyte", test_imgs_raw, test_count_raw, isz2))
        return false;
    if (!loadLabels(data_dir + "/t10k-labels-idx1-ubyte", test_lbls_raw, lbl_count))
        return false;

    image_size = isz1;

    // Combine all data then split 90/10 with fixed seed shuffle
    int total = train_count + test_count_raw;
    all_images.resize(total * image_size);
    all_labels.resize(total);

    std::copy(train_imgs.begin(), train_imgs.end(), all_images.begin());
    std::copy(test_imgs_raw.begin(), test_imgs_raw.end(), all_images.begin() + train_count * image_size);
    std::copy(train_lbls.begin(), train_lbls.end(), all_labels.begin());
    std::copy(test_lbls_raw.begin(), test_lbls_raw.end(), all_labels.begin() + train_count);

    // Shuffle with fixed seed
    std::vector<int> indices(total);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(42);
    std::shuffle(indices.begin(), indices.end(), rng);

    int num_train = static_cast<int>(total * 0.9);
    int num_test = total - num_train;

    dataset.image_size = image_size;
    dataset.num_train = num_train;
    dataset.num_test = num_test;
    dataset.train_images.resize(num_train * image_size);
    dataset.train_labels.resize(num_train);
    dataset.test_images.resize(num_test * image_size);
    dataset.test_labels.resize(num_test);

    for (int i = 0; i < num_train; i++) {
        int idx = indices[i];
        std::copy(all_images.begin() + idx * image_size,
                  all_images.begin() + (idx + 1) * image_size,
                  dataset.train_images.begin() + i * image_size);
        dataset.train_labels[i] = all_labels[idx];
    }

    for (int i = 0; i < num_test; i++) {
        int idx = indices[num_train + i];
        std::copy(all_images.begin() + idx * image_size,
                  all_images.begin() + (idx + 1) * image_size,
                  dataset.test_images.begin() + i * image_size);
        dataset.test_labels[i] = all_labels[idx];
    }

    printf("MNIST loaded: %d train, %d test, image size %d\n",
           num_train, num_test, image_size);

    return true;
}
