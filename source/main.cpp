#include <iostream>

#include <pkai/construction/network_builder.hpp>
#include <pkai/construction/connection/host/fully_connected.hpp>
#include <pkai/construction/allocators/host_allocator.hpp>

int main() {
    PKAI::NetworkBuilder
        ::DefineAllocator<PKAI::Allocators::HostAllocator>
        ::DefineFloatType<float>
        ::AddLayer<2>
        ::AddConnection<PKAI::Connection::Host::FullyConnected>
        ::AddLayer<2>
        ::Build network;

    float test[2] = { 1, 0 };

    network.set_inputs(test);
    network.activate();
    network.get_outputs(test);

    std::cout << "Out: " << test[0] << ", " << test[1] << "\n";
}