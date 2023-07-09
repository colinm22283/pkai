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

    {
        PKAI::HostDataset<float> dataset(25, 2);
        dataset.add({
                        1, 1, 1, 1, 1,
                        1, 0, 0, 0, 1,
                        1, 0, 0, 0, 1,
                        1, 0, 0, 0, 1,
                        1, 1, 1, 1, 1
                    }, {1, 0});
        dataset.add({
                        1, 1, 1, 0, 0,
                        1, 0, 1, 0, 0,
                        1, 0, 1, 0, 0,
                        1, 0, 1, 0, 0,
                        1, 1, 1, 0, 0
                    }, {1, 0});
        dataset.add({
                        0, 0, 1, 1, 1,
                        0, 0, 1, 0, 1,
                        0, 0, 1, 0, 1,
                        0, 0, 1, 0, 1,
                        0, 0, 1, 1, 1
                    }, {1, 0});
        dataset.add({
                        1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1
                    }, {0, 1});
        dataset.add({
                        1, 0, 1, 1, 1,
                        1, 0, 1, 0, 0,
                        1, 0, 1, 0, 0,
                        0, 0, 1, 1, 1,
                        1, 1, 1, 1, 1
                    }, {0, 1});

        dataset.save_to("training_data/loops.ds");
    }

    PKAI::HostDataset<float> dataset("training_data/loops.ds");

    PKAI::HostNetwork<
        PKAI::ActivationFunction::ReLu,
        float,
        PKAI::Layer<25>,
        PKAI::Host::FullyConnected,
        PKAI::Layer<50>,
        PKAI::Host::FullyConnected,
        PKAI::Layer<2>
    > network;

    float temp_buf[network.output_size()];

    for (int iter = 0; iter < 10000; iter++) {
        const auto & pair = dataset.get_random_pair();

        std::cout << "Input: [ ";
        for (int i = 0; i < network.input_size(); i++) {
            std::cout << pair.input()[i] << " ";
        }
        std::cout << "]\n";

        network.give_input(pair.input());

        network.activate();

        network.copy_output(temp_buf);
        network.backpropagate(pair.output());
        std::cout << "Output: [ ";
        for (int i = 0; i < network.output_size(); i++) {
            std::cout << network.output_ptr()[i] << " ";
        }
        std::cout << "]\n";
        std::cout << "Correct: [ ";
        for (int i = 0; i < network.output_size(); i++) {
            std::cout << pair.output()[i] << " ";
        }
        std::cout << "]\n";
    }

    std::cout << "Giving unique input...\n";

    float test_in[] = {
        0.5, 0.5, 0.5, 0, 0,
        0.5, 0,   0.5, 0, 0,
        0.5, 0,   0.5, 0, 0,
        0.5, 0,   0.5, 0, 0,
        0.5, 0.5, 0.5, 0, 0
    };

    network.give_input(test_in);
    network.activate();
    network.copy_output(temp_buf);
    std::cout << "Output: [ ";
    for (int i = 0; i < network.output_size(); i++) {
        std::cout << network.output_ptr()[i] << " ";
    }
    std::cout << "]\n";

    std::cout << "DONE!\n";

    return 0;
}
