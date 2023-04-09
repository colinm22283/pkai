#include <iostream>
#include <chrono>

#include <pkai/network.hpp>
#include <pkai/training_set.hpp>

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
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 65535);
    cudaDeviceSynchronize();

    check_cuda_error();

    PKAI::Network network2({
        2,
        10,
        10,
        1
    });
    network2.randomize(2854453934);
    check_cuda_error();

    network2.save("networks/xor.net");
    check_cuda_error();

    std::cout << "Loading network...\n";
    PKAI::Network network("networks/xor.net");

    std::cout << "Comparison\n";
    network2.print();
    network.print();
    check_cuda_error();
    return 0;

    std::cout << "Loading training set...\n";
    PKAI::TrainingSet training_set("training_data/xor.ts");

    std::cout << "Training...\n";

    auto start_time = std::chrono::high_resolution_clock::now();
    network.run(training_set, 300000, false, true, false);
    auto end_time = std::chrono::high_resolution_clock::now();
    check_cuda_error();

    unsigned long elapsed = (end_time - start_time).count();

    std::cout << "Training complete\n";
    printf(
        "Elapsed time: %2.lu:%2.lu:%2.3f\n\n",
        elapsed / 360000000000 % 60,
        elapsed / 60000000000 % 60,
        fmodf32((float) elapsed / 1000000000, 60)
    );
    std::cout << "Press enter to continue...\n";
    std::cin.get();

    network.run(training_set, training_set.size(), true, false, false);
    check_cuda_error();

    return 0;
}
