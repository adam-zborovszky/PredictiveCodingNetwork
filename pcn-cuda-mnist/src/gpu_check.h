#pragma once

#include <string>

struct GPUInfo {
    std::string name;
    size_t vram_bytes;
    int sm_major;
    int sm_minor;
    int device_id;
};

// Detects available CUDA GPUs, prints info, selects the one with most VRAM.
// Returns true if a GPU was found and selected, false otherwise.
bool detectAndSelectGPU(GPUInfo& selected);
