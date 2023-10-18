#include <pkai/host.hpp>
#include <pkai/universal/network_builder.hpp>
#include <pkai/universal/activation_function/relu.hpp>
#include <pkai/universal/connection/fully_connected.hpp>

int main() {
    using namespace PKAI;
    using namespace Connection;
    using namespace ActivationFunction;

    using Builder = PKAI::NetworkBuilder
        ::DefineFloatType<float>
        ::AddLayer<8>
        ::AddConnection<FullyConnected<ReLu>>
        ::AddLayer<20>
        ::AddConnection<FullyConnected<ReLu>>
        ::AddLayer<8>;

    static Builder::Network network;
    static Builder::Dataset dataset;

    dataset.emplace_set({
        {1, 0, 0, 0, 0, 0, 0, 0},
        {}
    });

    network.train<1000>(dataset, 1000000);

    float notes[8] = { 0, 0, 0, 0, 1, 0, 0, 0 };
    for (int i = 0; i < 10; i++) {
        network.set_inputs(notes);
        network.activate();

        float temp[4];
        network.get_outputs(temp);

        std::cout << "Outputs " << i << ": ";
        for (int j = 0; j < 4; j++) std::cout << std::round(temp[j]) << " ";
        std::cout << "\n";

        std::memcpy(&notes[0], &notes[4], 4 * sizeof(float));
        std::memcpy(&notes[4], temp, 4 * sizeof(float));
    }
}