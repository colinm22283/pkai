#include <pkai/network.hpp>

#include <cstdio>
#include <iostream>

#include <pkai/functions.hpp>
#include <pkai/indexers.hpp>

__global__
void activate_k(
    float ** neurons,
    float ** synapses,
    float ** biases,
    unsigned long layer,
    unsigned long layer_size,
    unsigned long from_size
) {
    const unsigned long idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= layer_size) return;

    neurons[layer][idx] = 0;

    for (int i = 0; i < from_size; i++) {
        PKAI::neuron(neurons, layer, idx) +=
            PKAI::synapse(synapses, layer - 1, layer_size, i, idx) *
            PKAI::neuron(neurons, layer - 1, i);
    }

    neurons[layer][idx] = PKAI::sigmoid(neurons[layer][idx] + biases[layer - 1][idx]);
//    neurons[layer][idx] = PKAI::fast_sigmoid(neurons[layer][idx] + biases[layer - 1][idx]);
//    neurons[layer][idx] = PKAI::tanh(neurons[layer][idx] + biases[layer - 1][idx]);
//    neurons[layer][idx] = PKAI::relu(neurons[layer][idx] + biases[layer - 1][idx]);
}

void PKAI::Network::activate() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    for (int i = 1; i < layer_count; i++) {
        activate_k<<<1, 1024, 0, stream>>>(
            neurons,
            synapses,
            biases,
            i,
            layer_sizes[i],
            layer_sizes[i - 1]
        );
    }

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}

/// \brief async version of the activate function
void PKAI::Network::activate(cudaStream_t stream) {
    for (int i = 1; i < layer_count; i++) {
        activate_k<<<1, 1024, 0, stream>>>(
            neurons,
            synapses,
            biases,
            i,
            layer_sizes[i],
            layer_sizes[i - 1]
        );
    }
}