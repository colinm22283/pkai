#include <iostream>
#include <chrono>

#include <pkai/network/host_network.hpp>
#include <pkai/connection/host/fully_connected.hpp>

#include <pkai/training/training_dataset.hpp>

#include <pkai/activation_function/relu.hpp>

#include <device/error_check.hpp>

int main(int argc, const char ** argv) {
//    cudaSetDevice(0);
//    cudaDeviceSetLimit(cudaLimitStackSize, 65536);
//    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 65536);
//    cudaDeviceSynchronize();
//    check_cuda_error();

    std::cout << "START\n";

    PKAI::HostDataset<float> dataset(2, 2);

    dataset.add({ 1, 0 }, { 0, 1 });
    dataset.add({ 0, 1 }, { 1, 0 });

    PKAI::HostNetwork<
        PKAI::ActivationFunction::ReLu,
        float,
        PKAI::Layer<2>,
        PKAI::Host::FullyConnected,
        PKAI::Layer<2>
    > network;

    float temp_buf[2];

    for (int i = 0; i < 10000; i++) {
        const auto & pair = dataset.get_random_pair();

        std::cout << "Input: [ " << pair.input()[0] << ", " << pair.input()[1] << " ]\n";

        network.give_input(pair.input());

        network.activate();

        network.copy_output(temp_buf);
        std::cout << "Output:  [ " << temp_buf[0] << ", " << temp_buf[1] << " ]\n";
        std::cout << "Correct: [ " << pair.output()[0] << ", " << pair.output()[1] << " ]\n\n";

        network.backpropagate(pair.output());
    }

    std::cout << "DONE!\n";

    return 0;
}
