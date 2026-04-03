#include "gpu_check.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

bool detectAndSelectGPU(GPUInfo& selected) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return false;
    }

    if (device_count == 0) {
        fprintf(stderr, "ERROR: No CUDA-capable GPU found.\n");
        fprintf(stderr, "Please ensure you have an NVIDIA GPU with CUDA support and the correct drivers installed.\n");
        return false;
    }

    printf("Found %d CUDA device(s):\n", device_count);

    int best_device = -1;
    size_t best_vram = 0;

    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        double vram_gb = static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0);
        printf("  [%d] %s — %.1f GB VRAM, SM %d.%d\n",
               i, prop.name, vram_gb, prop.major, prop.minor);

        if (prop.totalGlobalMem > best_vram) {
            best_vram = prop.totalGlobalMem;
            best_device = i;
        }
    }

    if (best_device < 0) {
        fprintf(stderr, "ERROR: Could not select a CUDA device.\n");
        return false;
    }

    cudaDeviceProp best_prop;
    cudaGetDeviceProperties(&best_prop, best_device);

    err = cudaSetDevice(best_device);
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaSetDevice(%d) failed: %s\n", best_device, cudaGetErrorString(err));
        return false;
    }

    selected.name = best_prop.name;
    selected.vram_bytes = best_prop.totalGlobalMem;
    selected.sm_major = best_prop.major;
    selected.sm_minor = best_prop.minor;
    selected.device_id = best_device;

    printf("Selected GPU: [%d] %s\n\n", best_device, best_prop.name);
    return true;
}
