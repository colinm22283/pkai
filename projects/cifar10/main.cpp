#include <iostream>
#include <fstream>
#include <algorithm>

#include <pkai/host.hpp>
#include <pkai/universal/connection/fully_connected.hpp>
#include <pkai/universal/activation_function/relu.hpp>
#include <pkai/universal/activation_function/leaky_relu.hpp>
#include <pkai/universal/activation_function/sigmoid.hpp>

#include "cifar10.hpp"

std::default_random_engine engine(149324023);
std::uniform_int_distribution distro(0, 50000 - 1);

inline float print_all(auto & dataset, auto & network) {
    constexpr int sample_size = 20;

    int correct_count = 0;
    float cost;

    for (PKAI::int_t i = 0; i < std::min(dataset.size(), (PKAI::int_t) sample_size); i++) {
        auto & set = dataset.get(distro(engine));

        float temp[set.out_size()];

        network.set_inputs(set.in());
        network.activate();
        network.get_outputs(temp);

//        std::cout << "In: " << set.in()[0];
//        for (int j = 1; j < set.in_size(); j++) std::cout << ", " << set.in()[j];
//        std::cout << "\n";
        std::cout << "Out: " << temp[0];
        for (int j = 1; j < set.out_size(); j++) std::cout << ", " << temp[j];
        std::cout << "\n";
        std::cout << "Correct: " << set.out()[0];
        for (int j = 1; j < set.out_size(); j++) std::cout << ", " << set.out()[j];
        std::cout << "\n";
        int correct = 0;
        for (int j = 1; j < set.out_size(); j++) if (set.out()[j] > set.out()[correct]) correct = j;
        std::cout << "Correct Index: " << correct << "\n";
        int greatest = network.greatest_output();
        std::cout << "Out Index: " << greatest << "\n";
        std::cout << "Certainty: " << temp[greatest] << "\n";
        float temp2 = network.cost(set.out());
        std::cout << "Cost: " << temp2 << "\n";
        if (correct == greatest) {
            std::cout << "Correct!\n";
            correct_count++;
        }
        else std::cout << "Incorrect :(\n";
        std::cout << "\n\n";

        cost += temp2;
    }

    std::cout << "Accuracy: " << (correct_count * 100 / std::min(dataset.size(), (PKAI::int_t) sample_size)) << "%\n";

    return cost / std::min(dataset.size(), (PKAI::int_t) sample_size);
}

inline void test_image(auto & network, const char * path) {
    std::ifstream test_image(path);
    float inputs[3072];
    float outputs[10];
    for (int i = 0; i < 1024; i++) {
        unsigned char temp[3];
        test_image.read((char *) &temp, 3 * sizeof(unsigned char));

        inputs[i] = (float) temp[0] / 256.0f;
        inputs[i + 1024] = (float) temp[1] / 256.0f;
        inputs[i + 2048] = (float) temp[2] / 256.0f;
    }

    network.set_inputs(inputs);
    network.activate();
    network.get_outputs(outputs);

    std::cout << "Outputs: " << outputs[0];
    for (int j = 1; j < 10; j++) std::cout << ", " << outputs[j];
    std::cout << "\n";
    int greatest = network.greatest_output();
    std::cout << "Out Index: " << greatest << "\n";
    std::cout << "Certainty: " << outputs[greatest] << "\n";
}

int main() {
    using namespace PKAI;
    using namespace Connection;
    using namespace ActivationFunction;

    using Builder = NetworkBuilder
        ::DefineFloatType<float>
        ::AddLayer<3072>
        ::AddConnection<FullyConnected<ReLu>>
        ::AddLayer<256>
        ::AddConnection<FullyConnected<ReLu>>
        ::AddLayer<10>;

    static Builder::NetworkType network;
    static Builder::DatasetType dataset("cifar10.ds");

//    Cifar10::load_datafile(dataset, "data_batch_1.bin");
//    Cifar10::load_datafile(dataset, "data_batch_2.bin");
//    Cifar10::load_datafile(dataset, "data_batch_3.bin");
//    Cifar10::load_datafile(dataset, "data_batch_4.bin");
//    Cifar10::load_datafile(dataset, "data_batch_5.bin");
//    dataset.save("cifar10.ds");

    std::cout << "Dataset size: " << dataset.size() << "\n";
    std::cout << "Starting training...\n";

    for (int i = 0; i < 20; i++) {
        network.train(dataset, 50000);
        float cost = print_all(dataset, network);

        std::cout << "\n" << i + 1 << "/1000" << ", Sample Cost: " << cost << "\n";

        if (i % 5 == 0) {
            std::cout << "Saving...\n";
            network.save("cifar10.net");
            std::cout << "Saved!\n\n";

//            std::cout << "Testing plane image\n";
//            test_image(network, "plane.data");
//            std::cout << "\n";
        }
    }

    std::cout << "Done!\n";

//    print_all(dataset, network);

    std::cout << "\nTotal Cost: " << network.total_cost(dataset) << "\n";
}