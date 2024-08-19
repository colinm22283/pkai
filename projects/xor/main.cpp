#include <iostream>

#include "pkai/host.hpp"
#include "pkai/universal/connection/fully_connected.hpp"

#include "pkai/universal/activation_function/step.hpp"
#include "pkai/universal/activation_function/linear.hpp"
#include "pkai/universal/activation_function/relu.hpp"
#include "pkai/universal/activation_function/sigmoid.hpp"

int main() {
    using namespace PKAI;
    using namespace Connection;
    using namespace ActivationFunction;

    using Builder = NetworkBuilder
        ::DefineFloatType<float>
        ::AddLayer<2>
        ::AddConnection<FullyConnected<Sigmoid>>
        ::AddLayer<8>
        ::AddConnection<FullyConnected<Sigmoid>>
        ::AddLayer<1>;

    Builder::NetworkType network;

    Builder::DatasetType dataset("xor.ds");

    network.train<10000>(dataset, 1000000);

    std::cout << "Training complete!\n\n";

    for (PKAI::int_t i = 0; i < dataset.size(); i++) {
        auto & set = dataset.get(i);

        float temp[1];

        network.set_inputs(set.in());
        network.activate();
        network.get_outputs(temp);

        std::cout << "In:      " << set.in()[0] << ", " << set.in()[1] << "\n";
        std::cout << "Out:     " << temp[0] << "\n";
        std::cout << "Correct: " << set.out()[0] << "\n";
    }

    std::cout << "\nTotal Cost: " << network.total_cost(dataset) << "\n";
}