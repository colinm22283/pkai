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

#define TRAINING_ITERATIONS 3000000

int main() {
    cudaSetDevice(0);
    cudaDeviceSetLimit(cudaLimitStackSize, 65536);
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 65535);
    cudaDeviceSynchronize();
    check_cuda_error();

    std::cout << "Loading network...\n";
    PKAI::Network network("networks/xor.net");
    check_cuda_error();

    std::cout << "Loading training set...\n";
    PKAI::TrainingSet training_set("training_data/xor.ts");
    check_cuda_error();

    std::cout << "Training...\n";

    auto start_time = std::chrono::high_resolution_clock::now();
    network.run(training_set, TRAINING_ITERATIONS, 500, [](PKAI::Network &, unsigned long iteration) {
        std::cout << iteration << '/' << TRAINING_ITERATIONS << '\n';
    });
    auto end_time = std::chrono::high_resolution_clock::now();
    check_cuda_error();

    unsigned long elapsed = (end_time - start_time).count();

    std::cout << "Training complete\n";
    printf(
        "Elapsed time: %2.lu:%2.lu:%2.3f\n\n",
        elapsed / 360000000000 % 60,
        elapsed / 60000000000 % 60,
        fmodf32((float) elapsed / 1000000000.0f, 60)
    );
    std::cout << "Press enter to continue...\n";
    std::cin.get();

    network.run(training_set, training_set.size(), [](PKAI::Network & network, PKAI::training_pair_t & current_pair, unsigned long iteration) {
        network.print();
        float temp[network.layer_sizes[network.layer_count - 1]];
        current_pair.extract_outputs(temp, network.layer_sizes[network.layer_count - 1]);

        std::cout << "Correct: [ ";
        for (unsigned long i = 0; i < network.layer_sizes[network.layer_count - 1]; i++) {
            std::cout << temp[0] << " ";
        }
        std::cout << "]\n";

    });
    check_cuda_error();

    return 0;
}
