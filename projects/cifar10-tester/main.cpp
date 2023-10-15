#include <pkai/host.hpp>
#include <pkai/universal/network_builder.hpp>
#include <pkai/universal/connection/fully_connected.hpp>
#include <pkai/universal/activation_function/relu.hpp>

int main() {
    using namespace PKAI;
    using namespace Connection;
    using namespace ActivationFunction;

    using Builder = PKAI::NetworkBuilder
        ::DefineFloatType<float>
        ::AddLayer<1024>
        ::AddConnection<FullyConnected<ReLu>>
        ::AddLayer<128>
        ::AddConnection<FullyConnected<ReLu>>
        ::AddLayer<64>
        ::AddConnection<FullyConnected<ReLu>>
        ::AddLayer<10>;

    static Builder::Network network("cifar10.net");
    static Builder::Dataset dataset("cifar10.ds");

    network.train<50000>(dataset, 1000000);

    std::ifstream fs("horse.data");

    float in[1024];

    for (int i = 0; i < 1024; i++) {
        unsigned char temp[3];
        fs.read((char *) &temp, 3 * sizeof(unsigned char));
        in[i] = ((float) temp[0] / 256.0f + (float) temp[1] / 256.0f + (float) temp[2] / 256.0f) / 3.0f;
        std::cout << in[i] << ", ";
    }
    std::cout << "\n";

    network.set_inputs(in);
    network.activate();
    std::cout << "Output: " << network.greatest_output() << "\n";

    float temp[10];
    network.get_outputs(temp);
    std::cout << "Out: " << temp[0];
    for (int j = 1; j < 10; j++) std::cout << ", " << temp[j];
    std::cout << "\n";

    network.save("cifar10.net");
}