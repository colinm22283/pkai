#pragma once

#include <cuda_runtime_api.h>

namespace PKAI {
    __device__
    static constexpr float & neuron(
        float ** const neurons,
        unsigned int layer,
        unsigned int pos
    ) {
        return neurons[layer][pos];
    }
    __device__
    static constexpr float & synapse(
        float ** const synapses,
        const unsigned int layer,
        const unsigned int to_size,
        const unsigned int from,
        const unsigned int to
    ) {
        return synapses[layer][to_size * from + to]; // swapped to and from??????
    }
    __device__
    static constexpr float & cost(
        float ** const costs,
        unsigned int layer,
        unsigned int pos
    ) {
        return costs[layer - 1][pos];
    }
    __device__
    static constexpr float & bias(
        float ** const biases,
        const unsigned int layer,
        const unsigned int pos
    ) {
        return biases[layer - 1][pos];
    }
}