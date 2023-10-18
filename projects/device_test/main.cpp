#include <pkai/device.hpp>
#include <pkai/universal/network_builder.hpp>
#include <pkai/universal/activation_function/relu.hpp>
#include <pkai/universal/connection/fully_connected.hpp>

int main() {
    using namespace PKAI;
    using namespace PKAI::ActivationFunction;
    using namespace PKAI::Connection;

    using Builder = PKAI::NetworkBuilder
        ::DefineFloatType<float>
        ::Layer<2>
        ::AddConnection<FullyConnected<ReLu>>
        ::Layer<2>;
}