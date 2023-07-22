#include <iostream>
#include <chrono>

#include <pkai/network/host_network.hpp>
#include <pkai/connection/host/fully_connected.hpp>
#include <pkai/connection/host/square_convolution.hpp>

#include <pkai/training/training_dataset.hpp>

#include <pkai/activation_function/relu.hpp>
#include <pkai/activation_function/sigmoid.hpp>
#include <pkai/activation_function/tanh.hpp>

#include <device/error_check.hpp>

int main() {
//    cudaSetDevice(0);
//    cudaDeviceSetLimit(cudaLimitStackSize, 65536);
//    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 65536);
//    cudaDeviceSynchronize();
//    check_cuda_error();

    std::cout << "START\n";

//    PKAI::HostDataset<float> dataset(4, 2);
//    dataset.add({ 1, 0 }, { 0, 1 });
//    dataset.add({ 0, 1 }, { 1, 0 });

//    dataset.add({ 1, 0, 1, 0 }, { 0, 1 });
//    dataset.add({ 1, 0, 0, 1 }, { 1, 0 });
//    dataset.add({ 0, 1, 1, 0 }, { 1, 0 });
//    dataset.add({ 0, 1, 0, 1 }, { 0, 1 });

    PKAI::HostDataset<float> dataset("training_data/cifar10.ds");

    std::cout << "Data size: " << dataset.size() << "\n";

    PKAI::HostNetwork<
        PKAI::ActivationFunction::ReLu,
        float,
        PKAI::Layer<3072>,
        PKAI::Host::FullyConnected,
        PKAI::Layer<2000>,
        PKAI::Host::FullyConnected,
        PKAI::Layer<1000>,
        PKAI::Host::FullyConnected,
        PKAI::Layer<10>
    > network;

    network.load("networks/cifar10.net");

    static constexpr int total_iter = 10000000;
    for (int iter = 0; iter < total_iter; iter++) {
        if (iter % 500 == 0) {
            int correct_count = 0;
            float temp_buf[network.output_size()];
            for (int iter2 = 0; iter2 < 200; iter2++) {
                const auto & pair2 = dataset.get_random_pair();

//        std::cout << "Input: [ ";
//        for (int i = 0; i < network.input_size(); i++) {
//            std::cout << pair.input()[i] << " ";
//        }
//        std::cout << "]\n";

                network.give_input(pair2.input());

                network.activate();

                network.copy_output(temp_buf);
                network.backpropagate(pair2.output());
                std::cout << "Output: [ ";
                for (int i = 0; i < network.output_size(); i++) {
                    std::cout << network.output_ptr()[i] << " ";
                }
                std::cout << "]\n";
                std::cout << "Correct: [ ";
                for (int i = 0; i < network.output_size(); i++) {
                    std::cout << pair2.output()[i] << " ";
                }
                std::cout << "]\n";
                std::cout << "Average cost: " << network.average_cost(pair2.output()) << "\n";

                int best_index = 0;
                for (int i = 0; i < network.output_size(); i++) {
                    if (network.output_ptr()[i] > network.output_ptr()[best_index]) {
                        best_index = i;
                    }
                }
                int correct_index = 0;
                for (int i = 0; i < network.output_size(); i++) {
                    if (pair2.output()[i] > pair2.output()[correct_index]) {
                        correct_index = i;
                    }
                }

                std::cout << "Indexes:\n";
                std::cout << "\tResult: " << best_index << "\n";
                std::cout << "\tCorrect: " << correct_index << "\n";
                if (best_index == correct_index) {
                    std::cout << "\t-------------------------------------------------------------- CORRECT!\n";
                    correct_count++;
                }
            }

            std::cout << "Accuracy: " << (float) correct_count * 0.5f << "%\n";
            std::ofstream correct_log("correct_log.csv", std::ofstream::app);
            correct_log << "," << std::to_string((float) correct_count * 0.5f);

            std::cout << "Saving...\n";
            network.save("networks/cifar10.net");
        }

        const auto & pair = dataset.get_random_pair();

        network.give_input(pair.input());
        network.activate();
        network.backpropagate(pair.output());

        if (iter % 250 == 0) std::cout << iter << "/" << total_iter << '\n';
    }

    std::cout << "Saving...\n";
    network.save("networks/cifar10.net");

    std::cout << "DONE!\n";

    return 0;
}
