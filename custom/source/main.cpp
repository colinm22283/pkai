#include <iostream>
#include <chrono>
#include <fstream>

#include "pkai/network.hpp"
#include "pkai/training_set.hpp"

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

    std::cout << "Loading network...\n";
    PKAI::Network network(network_path);
    check_cuda_error();

    std::cout << "Loading training set...\n";
    PKAI::TrainingSet training_set(dataset_path);
    check_cuda_error();

    std::cout << "Training...\n";

    auto start_time = std::chrono::high_resolution_clock::now();
    network.run(training_set, training_iterations, 1000, [](PKAI::Network & network, unsigned long iteration) {
        std::cout << iteration << '/' << training_iterations << '\n';
        std::cout << "Saving network...\n";
        network.save(network_path);
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

    network.run(training_set, 50, [](PKAI::Network & network, PKAI::training_pair_t & current_pair, unsigned long iteration) {
//        network.print();
        float temp[network.layer_sizes[network.layer_count - 1]];

        network.extract_outputs(temp);

        std::cout << "Output: [ ";
        for (unsigned long i = 0; i < network.layer_sizes[network.layer_count - 1]; i++) {
            std::cout << temp[i] << " ";
        }
        std::cout << "]\n";

        current_pair.extract_outputs(temp, network.layer_sizes[network.layer_count - 1]);

        std::cout << "Correct: [ ";
        for (unsigned long i = 0; i < network.layer_sizes[network.layer_count - 1]; i++) {
            std::cout << temp[i] << " ";
        }
        std::cout << "]\n";

    });
    check_cuda_error();

    std::cout << "Saving network...\n";
    network.save(network_path);
    check_cuda_error();

    std::cout << "Done!\n";

    return 0;
}
