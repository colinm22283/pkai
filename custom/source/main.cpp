#include <iostream>
#include <chrono>
#include <fstream>

#include <cuda_runtime_api.h>

constexpr unsigned long training_iterations = 51000;
constexpr const char * network_path = "networks/cifar10.net";
constexpr const char * dataset_path = "training_data/cifar10.ts";

inline void check_cuda_error() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;

        exit(1);
    }
}

int main() {
    cudaSetDevice(0);
    cudaDeviceSetLimit(cudaLimitStackSize, 65536);
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 268431360);
    cudaDeviceSynchronize();
    check_cuda_error();



    return 0;
}
